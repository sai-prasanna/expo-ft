#!/usr/bin/env bash
# Supervised fine-tuning of pi0.5 (LoRA) on the DK1 demos. LoRA needs > 22.5 GB GPU memory.
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ID="${DATA_ID:-dk1_task_20}"
REPO_ID="expo_ft/${DATA_ID}"
ASSETS_DIR="./assets/expo_pi05_dk1_lora_finetune_sft"
ASSET_ID="expo_ft/${DATA_ID}"
# Config default is 64, but that peaks at ~37 GB and JAX gets 75% of the GPU (~24 GB on a 32 GB RTX 5090),
# so batch 32 here. For batch 64 use an 80 GB GPU: scripts/dk1_task/finetune_cluster.sh.
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_STEPS="${NUM_STEPS:-4001}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"

python expo_ft/agents/vla/openpi/scripts/train.py expo_pi05_dk1_lora_finetune_sft \
    --exp-name="${DATA_ID}_lora_sft" \
    --resume \
    --data.repo_id="$REPO_ID" \
    --data.assets.assets_dir="$ASSETS_DIR" \
    --data.assets.asset_id="$ASSET_ID" \
    --batch-size="$BATCH_SIZE" \
    --num_train_steps="$NUM_STEPS" \
    --save_interval="$SAVE_INTERVAL" \
    --fsdp_devices=1
