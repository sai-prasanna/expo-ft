#!/usr/bin/env bash
# Rollout client on the robot machine (client_dk1 venv). It dials the learner/eval server.
# Interventions: HOLD the foot pedal (spare mouse) to take over with the leader arms, release to
# hand back to the policy. The client opens the pedal itself.
# Outcome (pedal UP): open both leader grippers, then squeeze + hold the LEFT one = SUCCESS,
# the RIGHT one = FAILURE. Same at the step limit. Keyboard fallback: 1 / 2 + Enter here.
source client_dk1/.venv/bin/activate

# TODO(lab): GPU server hostname. localhost = client and server on the same machine.
SERVER_HOST="${SERVER_HOST:-localhost}"
TASK_CONFIG="${TASK_CONFIG:-configs/task/dk1_task.py}"

python -m client.run_client \
    --host="$SERVER_HOST" \
    --port=8102 \
    --config_task_path="$TASK_CONFIG"
