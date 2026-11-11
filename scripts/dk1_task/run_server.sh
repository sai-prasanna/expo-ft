#!/usr/bin/env bash
# EXPO-FT learner (non-real-time, --delay=0). Hyperparameters as in scripts/pick/run_server.sh.
# edit_scale 0.05, not upstream's 0.2: our actions are absolute joint targets (upstream: Cartesian
# velocities), and per-step edits at 0.2 gave 0.065 rad mean error and 10-16% over-cap steps on
# replayed demos vs 0.02 rad and 1-7% at 0.05 (no edits: 0.01 rad, 2-4%).
# Robot client, no interventions:  TASK_CONFIG=configs/task/dk1_task_rl.py bash scripts/dk1_task/run_policy.sh
# Extra flags pass through; CLIENT_PORT / RUN_NAME override the port and run name.
# MAX_STEPS = env steps AND replay capacity (preallocated, ~460 KB/step for 3 cameras; 500k OOMs on 62 GB).
[ -n "${VIRTUAL_ENV:-}" ] || source .venv/bin/activate

# The learner listens here; the rollout client (run_policy.sh) dials in.
CLIENT_IP=0.0.0.0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# JAX takes 75% of the GPU by default; the first update (64 x utd 20 batches, 3 prefetched)
# OOMs at that on the 32 GB 5090.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"

# Pick-into-bowl: TASK_CONFIG=configs/task/dk1_pick.py DATA_ID=dk1_pnp_bowl_new_50 NUM_DATA=50
TASK_CONFIG="${TASK_CONFIG:-configs/task/dk1_task.py}"
DATA_ID="${DATA_ID:-dk1_task_20}"
NUM_DATA="${NUM_DATA:-20}"
SFT_CKPT="${SFT_CKPT:-./checkpoints/expo_pi05_dk1_lora_finetune_sft/${DATA_ID}_lora_sft/4000/params}"
DATASET_PATH="${DATASET_PATH:-./data/stack_green_blue/success}"
ASSETS_DIR="${ASSETS_DIR:-./assets/expo_pi05_dk1_lora_finetune_sft}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/dk1_task}"

python train_pi_robo.py \
    --config_task="$TASK_CONFIG" \
    --dataset_path="$DATASET_PATH" \
    --num_data="$NUM_DATA" \
    --update_type=episode \
    --num_updates=3 \
    --offline_ratio=0 \
    --replan_steps=8 \
    --config=configs/model/expo_ft_pi_config.py \
    --config.N=8 \
    --config.n_edit_samples=8 \
    --config.edit_scale=0.05 \
    --config.pi05_config_name=expo_pi05_dk1_lora_finetune_sft \
    --config.pi05_weight_loader_path="$SFT_CKPT" \
    --config.pi05_assets_dir="$ASSETS_DIR" \
    --config.pi05_asset_id="expo_ft/${DATA_ID}" \
    --project_name=expo_ft_dk1 \
    --output_dir="$OUTPUT_DIR" \
    --client_host="$CLIENT_IP" \
    --client_port="${CLIENT_PORT:-8102}" \
    --max_steps="${MAX_STEPS:-40000}" \
    --fsdp_devices=1 \
    --resume \
    --checkpoint_model \
    --checkpoint_buffer \
    --checkpoint_interval=2000 \
    --run_name="${RUN_NAME:-expo_dk1_task}" \
    "$@"
