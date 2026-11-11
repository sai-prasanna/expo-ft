"""dk1_pick served from recorded demos (data/dk1_pnp_bowl_new), for hardware-free pipeline checks.
See configs/task/dk1_replay.py; use a non-robot port."""

from client.envs.dk1_replay_env import DK1ReplayEnv
from configs.task import dk1_pick


def get_config():
    config = dk1_pick.get_config()
    config.env_name = "dk1_pick_replay"
    config.env = DK1ReplayEnv
    config.replay_data_dir = "data/dk1_pnp_bowl_new/success"
    config.replay_episodes = (0, 7, 15)
    config.replay_log = ""
    return config
