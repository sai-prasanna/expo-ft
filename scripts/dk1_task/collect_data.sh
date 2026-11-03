#!/usr/bin/env bash
# Teleoperated demos with the leader arms (client_dk1 venv). Type 1+Enter = success, 2+Enter = discard.
# Reruns with the same SAVE_ROOT add episodes. One folder per task, e.g.
#   SAVE_ROOT=data/dk1_pick PROMPT="pick up the cube" bash scripts/dk1_task/collect_data.sh
source client_dk1/.venv/bin/activate

NUM_EPISODES="${NUM_EPISODES:-20}"
SAVE_ROOT="${SAVE_ROOT:-data/dk1_task}"
PROMPT="${PROMPT:-}"  # empty = language_instruction from the task config

python -m client.collect_data_dk1 \
    --save_root "$SAVE_ROOT" \
    --num_episodes "$NUM_EPISODES" \
    --prompt "$PROMPT" \
    --task_config configs/task/dk1_task.py
