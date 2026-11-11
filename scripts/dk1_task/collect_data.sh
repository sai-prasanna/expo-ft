#!/usr/bin/env bash
# Teleoperated demos with the leader arms (client_dk1 venv). CLICK the foot pedal = start recording, CLICK = stop,
# then open both leader grippers and squeeze + hold LEFT = keep, RIGHT = discard (1/2+Enter also work).
# Old home-and-hold-still stop: add --nopedal_control.
# Reruns with the same SAVE_ROOT add episodes. One folder per task, e.g.
#   SAVE_ROOT=data/dk1_pick TASK_CONFIG=configs/task/dk1_pick.py NUM_EPISODES=40 bash scripts/dk1_task/collect_data.sh
source client_dk1/.venv/bin/activate

NUM_EPISODES="${NUM_EPISODES:-20}"
SAVE_ROOT="${SAVE_ROOT:-data/dk1_task}"
PROMPT="${PROMPT:-}"  # empty = language_instruction from the task config
TASK_CONFIG="${TASK_CONFIG:-configs/task/dk1_task.py}"

python -m client.collect_data_dk1 \
    --save_root "$SAVE_ROOT" \
    --num_episodes "$NUM_EPISODES" \
    --prompt "$PROMPT" \
    --task_config "$TASK_CONFIG"
