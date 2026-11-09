#!/usr/bin/env bash
# Evaluate the SFT policy (default) or an EXPO-FT checkpoint.
#
# Without an RL checkpoint, eval_droid_policy.py uses base (SFT) actions only: the critic and
# edit actor would be untrained. With an RL checkpoint it uses the full EXPO selection. Extra
# flags are passed through, e.g.
#   bash scripts/dk1_task/eval_policy.sh --checkpoint_dir=./checkpoints/dk1_task/expo_dk1_task/checkpoints \
#       --checkpoint_step=<step>        # add --only_base_actions to compare against the base policy
source .venv/bin/activate
CLIENT_IP=0.0.0.0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Another task, e.g. pick-into-bowl:
#   DATA_ID=dk1_pnp_blue_bowl_20 TASK_CONFIG=configs/task/dk1_pick.py DATASET_PATH=./data/dk1_pnp_blue_bowl/success \
#   SFT_CKPT=./checkpoints/expo_pi05_dk1_lora_finetune_sft/dk1_pnp_blue_bowl_20_lora_sft/2000/params bash scripts/dk1_task/eval_policy.sh
DATA_ID="${DATA_ID:-dk1_task_20}"
SFT_CKPT="${SFT_CKPT:-./checkpoints/expo_pi05_dk1_lora_finetune_sft/${DATA_ID}_lora_sft/4000/params}"
TASK_CONFIG="${TASK_CONFIG:-configs/task/dk1_task.py}"
DATASET_PATH="${DATASET_PATH:-./data/stack_green_blue/success}"
NUM_EPISODES="${NUM_EPISODES:-20}"

python eval_droid_policy.py \
    --config_task="$TASK_CONFIG" \
    --config=configs/model/expo_ft_pi_config.py \
    --dataset_path="$DATASET_PATH" \
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
    "$@"
