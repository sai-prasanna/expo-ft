#!/usr/bin/env bash
# EXPO-FT learner on a cluster GPU; the DK1 robot client connects through an SSH tunnel.
#
#   cluster submit ferranti --gpus 1 --time 08:00:00 --sbatch=--mem=96G -n dk1-rl-nohil \
#       -m "..." -- bash scripts/dk1_task/run_server_cluster.sh
#
# Then, on the robot PC (node name is printed in the job log as "=== learner node: ... ==="):
#   ssh -N -L 8102:<node>:8102 ferranti
#   TASK_CONFIG=configs/task/dk1_task_rl.py bash scripts/dk1_task/run_policy.sh
#
# Checkpoints and the replay buffer go to a fixed directory under CLUSTER_SHARED_DIR, so a new
# job with the same RUN_NAME resumes (--resume) where the last one stopped.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/weka/martius/mot244/data/expo-ft}"
export SFT_CKPT="${SFT_CKPT:-/weka/martius/mot244/runs/expo-ft/20261104-1616-dk1-sft-bs64-95333eeb/ckpt/expo_pi05_dk1_lora_finetune_sft/dk1_task_20_lora_sft/4000/params}"
export DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/raw/stack_green_blue/success}"
export ASSETS_DIR="$DATA_ROOT/assets/expo_pi05_dk1_lora_finetune_sft"
export RUN_NAME="${RUN_NAME:-expo_dk1_task_nohil}"
export OUTPUT_DIR="${OUTPUT_DIR:-${CLUSTER_SHARED_DIR:-$PWD/.cluster-shared}/rl}"
export CLIENT_PORT="${CLIENT_PORT:-8102}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

export OPENPI_DATA_HOME="$DATA_ROOT/openpi"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/weka/martius/mot244/uvcache}"
export UV_PROJECT_ENVIRONMENT="${CLUSTER_SHARED_DIR:-$PWD/.cluster-shared}/venv"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${CLUSTER_OUTPUT_DIR:-$OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR" "$WANDB_DIR"

"$HOME/.local/bin/uv" sync --frozen
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
nvidia-smi -L
echo "=== learner node: $(hostname)  port: $CLIENT_PORT ==="
echo "=== robot PC: ssh -N -L $CLIENT_PORT:$(hostname):$CLIENT_PORT ferranti ==="

bash scripts/dk1_task/run_server.sh --notqdm "$@"
