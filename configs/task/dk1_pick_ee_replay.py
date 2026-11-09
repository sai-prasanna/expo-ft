"""dk1_pick_ee served from recorded demos (hardware-free pipeline check; use a non-robot port).
The replay env's per-episode summary compares the IK'd joint commands with the demo's joint actions."""

from client.envs.dk1_ee import make_ee_env
from client.envs.dk1_replay_env import DK1ReplayEnv
from configs.task import dk1_pick_ee


def get_config():
    config = dk1_pick_ee.get_config()
    config.env_name = "dk1_pick_ee_replay"
    config.env = make_ee_env(DK1ReplayEnv)
    config.replay_data_dir = "data/dk1_pnp_bowl_new/success"
    config.replay_episodes = (0, 17, 33)
    config.replay_log = ""
    return config
