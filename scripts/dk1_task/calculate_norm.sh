#!/usr/bin/env bash
source .venv/bin/activate

REPO_ID="expo_ft/dk1_task_20"

python expo_ft/agents/vla/openpi/scripts/compute_norm_stats.py \
    --config-name expo_pi05_dk1_lora_finetune_sft \
    --repo-id "$REPO_ID"
