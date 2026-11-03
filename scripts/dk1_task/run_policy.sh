#!/usr/bin/env bash
# Rollout client on the robot machine (client_dk1 venv). It dials the learner/eval server.
# Interventions: run `python -m client.teleop.clutch` in a SECOND terminal (client_dk1 venv)
# and press SPACE to take over with the leader arms / hand back to the policy.
# Success / reset: type 1 or 2 + Enter in THIS terminal.
source client_dk1/.venv/bin/activate

# TODO(lab): GPU server hostname. localhost = client and server on the same machine.
SERVER_HOST="${SERVER_HOST:-localhost}"
TASK_CONFIG="${TASK_CONFIG:-configs/task/dk1_task.py}"

python -m client.run_client \
    --host="$SERVER_HOST" \
    --port=8102 \
    --config_task_path="$TASK_CONFIG"
