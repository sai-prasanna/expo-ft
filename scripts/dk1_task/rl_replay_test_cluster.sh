#!/usr/bin/env bash
# Hardware-free check of the EXPO-FT learner on one cluster GPU: the learner and a replay client
# (recorded demo frames, client/envs/dk1_replay_env.py) run together on the node. Verifies the
# update fits and completes at the default batch 64 / utd 20 / actor 64, and
# prints per-episode [replay] motion stats.
#
#   cluster submit ferranti --gpus 1 --time 01:30:00 --sbatch=--mem=96G -n dk1-rl-replay-test \
#       -m "..." -- bash scripts/dk1_task/rl_replay_test_cluster.sh
#
# Expects under DATA_ROOT: raw/stack_green_blue/success/<id>/traj.hdf5, assets/, openpi/.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/weka/martius/mot244/data/expo-ft}"
export SFT_CKPT="${SFT_CKPT:-/weka/martius/mot244/runs/expo-ft/20261104-1616-dk1-sft-bs64-95333eeb/ckpt/expo_pi05_dk1_lora_finetune_sft/dk1_task_20_lora_sft/4000/params}"
export DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/raw/stack_green_blue/success}"
export ASSETS_DIR="$DATA_ROOT/assets/expo_pi05_dk1_lora_finetune_sft"
export OUTPUT_DIR="${CLUSTER_OUTPUT_DIR:-./outputs/rl_replay_test}"
export MAX_STEPS="${MAX_STEPS:-12000}"   # replay capacity too: 5059 demo steps + ~25 episodes
export CLIENT_PORT="${CLIENT_PORT:-$((20000 + RANDOM % 20000))}"
export RUN_NAME=rl_replay_test
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

export OPENPI_DATA_HOME="$DATA_ROOT/openpi"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/weka/martius/mot244/uvcache}"
export UV_PROJECT_ENVIRONMENT="${CLUSTER_SHARED_DIR:-$PWD/.cluster-shared}/venv"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

"$HOME/.local/bin/uv" sync --frozen
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
nvidia-smi -L

# The replay config reads REPLAY_DIR (data/<name>/success) relative to the repo.
REPLAY_CONFIG="${REPLAY_CONFIG:-configs/task/dk1_replay.py}"
REPLAY_DIR="${REPLAY_DIR:-data/stack_green_blue/success}"
mkdir -p "$(dirname "$REPLAY_DIR")"
ln -sfn "$DATASET_PATH" "$REPLAY_DIR"

python -m client.run_client --host=localhost --port="$CLIENT_PORT" \
    --config_task_path="$REPLAY_CONFIG" > "$OUTPUT_DIR/replay_client.log" 2>&1 &
CLIENT_PID=$!
trap 'kill $CLIENT_PID 2>/dev/null || true' EXIT

bash scripts/dk1_task/run_server.sh \
    --nocheckpoint_model --nocheckpoint_buffer --notqdm "$@"

echo "=== replay episodes ==="
grep '^\[replay\]' "$OUTPUT_DIR/replay_client.log" | nl
