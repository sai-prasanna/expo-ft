#!/usr/bin/env bash
# LoRA SFT of pi0.5 on the DK1 demos, on a SLURM cluster via the `cluster` CLI:
#
#   cluster submit ferranti --gpus 2 --time 06:00:00 -n dk1-sft-bs64 -m "..." \
#       -- bash scripts/dk1_task/finetune_cluster.sh
#
# Expects pre-staged under DATA_ROOT (rsync from the GPU workstation):
#   lerobot/expo_ft/<DATA_ID>/                              LeRobot dataset (convert_data.sh)
#   assets/expo_pi05_dk1_lora_finetune_sft/expo_ft/<DATA_ID>/norm_stats.json  (calculate_norm.sh)
#   openpi/openpi-assets/checkpoints/pi05_base/            base weights (~/.cache/openpi)
# The venv lives in $CLUSTER_SHARED_DIR/venv and is re-synced against this run's code snapshot,
# so the editable openpi install points at this run's (vendored) openpi.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/weka/martius/mot244/data/expo-ft}"
DATA_ID="${DATA_ID:-dk1_task_20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_STEPS="${NUM_STEPS:-4001}"

export HF_LEROBOT_HOME="$DATA_ROOT/lerobot"
export OPENPI_DATA_HOME="$DATA_ROOT/openpi"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/weka/martius/mot244/uvcache}"
export UV_PROJECT_ENVIRONMENT="${CLUSTER_SHARED_DIR:-$PWD/.cluster-shared}/venv"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${CLUSTER_OUTPUT_DIR:-./outputs}"
CKPT_BASE="${CLUSTER_CKPT_DIR:-./checkpoints}"
mkdir -p "$WANDB_DIR" "$CKPT_BASE"

"$HOME/.local/bin/uv" sync --frozen
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
nvidia-smi -L

python expo_ft/agents/vla/openpi/scripts/train.py expo_pi05_dk1_lora_finetune_sft \
    --exp-name="${DATA_ID}_lora_sft" \
    --resume \
    --checkpoint-base-dir="$CKPT_BASE" \
    --data.repo_id="expo_ft/${DATA_ID}" \
    --data.assets.assets_dir="$DATA_ROOT/assets/expo_pi05_dk1_lora_finetune_sft" \
    --data.assets.asset_id="expo_ft/${DATA_ID}" \
    --batch-size="$BATCH_SIZE" \
    --num-workers=8 \
    --num_train_steps="$NUM_STEPS" \
    --save_interval=2000 \
    --fsdp_devices=1
