#!/usr/bin/env bash
# Evaluate the SFT policy (default) or an EXPO-FT checkpoint.
#
# SFT eval runs with --only_base_actions: with no RL checkpoint the EXPO critic and edit actor
# are randomly initialised, so "full" mode would pick among randomly edited chunks with a
# random Q (on 2026-11-04 that made the arms thrash). Extra flags are passed through, e.g.
#   bash scripts/dk1_task/eval_policy.sh --checkpoint_dir=./checkpoints/dk1_task/expo_dk1_task/checkpoints \
#       --checkpoint_step=<step>        # RL checkpoint; set ONLY_BASE=0 to use the critic/edits
source .venv/bin/activate
CLIENT_IP=0.0.0.0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ID="dk1_task_20"
SFT_CKPT="./checkpoints/expo_pi05_dk1_lora_finetune_sft/${DATA_ID}_lora_sft/4000/params"
NUM_EPISODES="${NUM_EPISODES:-20}"
ONLY_BASE="${ONLY_BASE:-1}"
BASE_FLAG=$([ "$ONLY_BASE" = "1" ] && echo "--only_base_actions" || echo "--noonly_base_actions")

python eval_droid_policy.py \
    --config_task=configs/task/dk1_task.py \
    --config=configs/model/expo_ft_pi_config.py \
    --dataset_path=./data/stack_green_blue/success \
    --num_data=1 \
    --client_host="$CLIENT_IP" \
    --client_port=8102 \
    --config.N=8 \
    --config.n_edit_samples=8 \
    --config.edit_scale=0.2 \
    --config.pi05_config_name=expo_pi05_dk1_lora_finetune_sft \
    --config.pi05_weight_loader_path="$SFT_CKPT" \
    --config.pi05_assets_dir="./assets/expo_pi05_dk1_lora_finetune_sft" \
    --config.pi05_asset_id="expo_ft/${DATA_ID}" \
    --num_episodes="$NUM_EPISODES" \
    "$BASE_FLAG" \
    "$@"
