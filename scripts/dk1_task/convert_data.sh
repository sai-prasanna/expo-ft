#!/usr/bin/env bash
# traj.hdf5 -> LeRobot (server venv, openpi's pinned LeRobot). The prompt comes from the task config.
source .venv/bin/activate

# Override for another task, e.g. pick-into-bowl:
#   MAX_EPISODES=40 TASK_CONFIG=configs/task/dk1_pick.py DATA_DIR=./data/dk1_pick/success REPO_NAME=expo_ft/dk1_pick_40 bash scripts/dk1_task/convert_data.sh
MAX_EPISODES="${MAX_EPISODES:-20}"
TASK_CONFIG="${TASK_CONFIG:-configs/task/dk1_task.py}"
DATA_DIR="${DATA_DIR:-./data/stack_green_blue/success}"
REPO_NAME="${REPO_NAME:-expo_ft/dk1_task_${MAX_EPISODES}}"
# joint (absolute joint targets) or ee_cmd (EE steps relative to the previous command, expo_ft/env/dk1_ee.py), e.g.
#   ACTION_SPACE=ee_cmd MAX_EPISODES=50 TASK_CONFIG=configs/task/dk1_pick.py DATA_DIR=./data/dk1_pnp_bowl_new/success \
#   REPO_NAME=expo_ft/dk1_pnp_bowl_new_eecmd_50 bash scripts/dk1_task/convert_data.sh
ACTION_SPACE="${ACTION_SPACE:-joint}"

uv run scripts/convert_dk1_data_to_lerobot.py \
    --data_dir="$DATA_DIR" \
    --repo_name="$REPO_NAME" \
    --task_config="$TASK_CONFIG" \
    --max_episodes="$MAX_EPISODES" \
    --action_space="$ACTION_SPACE" \
    --no-push-to-hub
