#!/usr/bin/env bash
# traj.hdf5 -> LeRobot (server venv, openpi's pinned LeRobot). The prompt comes from the task config.
source .venv/bin/activate

MAX_EPISODES=20
TASK_CONFIG="configs/task/dk1_task.py"
DATA_DIR="./data/stack_green_blue/success"
REPO_NAME="expo_ft/dk1_task_${MAX_EPISODES}"

uv run scripts/convert_dk1_data_to_lerobot.py \
    --data_dir="$DATA_DIR" \
    --repo_name="$REPO_NAME" \
    --task_config="$TASK_CONFIG" \
    --max_episodes="$MAX_EPISODES" \
    --no-push-to-hub
