"""DK1 task served from recorded demos instead of hardware (client/envs/dk1_replay_env.py).

Use with the eval/learner server on a NON-robot port, e.g.:
  NUM_EPISODES=3 bash scripts/dk1_task/eval_policy.sh --client_port=8199 --nosave_video
  client_dk1/.venv/bin/python -m client.run_client --host=localhost --port=8199 \
      --config_task_path=configs/task/dk1_replay.py
"""

from client.envs.dk1_replay_env import DK1ReplayEnv
from configs.task import dk1_task


def get_config():
    config = dk1_task.get_config()
    config.env_name = "dk1_replay"
    config.env = DK1ReplayEnv
    config.replay_data_dir = "data/stack_green_blue/success"
    config.replay_episodes = (0, 7, 15)
    config.replay_log = ""
    return config
