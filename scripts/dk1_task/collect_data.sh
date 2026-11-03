#!/usr/bin/env bash
# Teleoperated demos with the leader arms (client_dk1 venv). Type 1+Enter = success, 2+Enter = discard.
source client_dk1/.venv/bin/activate

NUM_EPISODES=20

python -m client.collect_data_dk1 \
    --save_root data/dk1_task \
    --num_episodes $NUM_EPISODES \
    --task_config configs/task/dk1_task.py
