#!/usr/bin/env bash
# EXPO-FT learner (non-real-time, --delay=0). Hyperparameters as in scripts/pick/run_server.sh.
source .venv/bin/activate

# The learner listens here; the rollout client (run_policy.sh) dials in.
CLIENT_IP=0.0.0.0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ID="dk1_task_20"
SFT_CKPT="./checkpoints/expo_pi05_dk1_lora_finetune_sft/${DATA_ID}_lora_sft/4000/params"

python train_pi_robo.py \
    --config_task=configs/task/dk1_task.py \
    --dataset_path=./data/stack_green_blue/success \
    --num_data=20 \
    --update_type=episode \
    --num_updates=3 \
    --offline_ratio=0 \
    --replan_steps=8 \
    --config=configs/model/expo_ft_pi_config.py \
    --config.N=8 \
    --config.n_edit_samples=8 \
    --config.edit_scale=0.2 \
    --config.pi05_config_name=expo_pi05_dk1_lora_finetune_sft \
    --config.pi05_weight_loader_path="$SFT_CKPT" \
    --config.pi05_assets_dir="./assets/expo_pi05_dk1_lora_finetune_sft" \
    --config.pi05_asset_id="expo_ft/${DATA_ID}" \
    --project_name=expo_ft_dk1 \
    --output_dir=./checkpoints/dk1_task \
    --client_host="$CLIENT_IP" \
    --client_port=8102 \
    --fsdp_devices=1 \
    --resume \
    --checkpoint_model \
    --checkpoint_buffer \
    --checkpoint_interval=2000 \
    --run_name=expo_dk1_task
