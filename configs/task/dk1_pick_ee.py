"""dk1_pick with a DROID-style end-effector action space (client/envs/dk1_ee.py).

Per arm: action = [dpos/0.075 m, deuler/0.15 rad (base frame), gripper 0..1], state = [pos, rot6d,
gripper] (14-D action, 20-D state). The env IKs each action to joints (mink, seeded at the measured
joints), then applies the same joint safety as dk1_pick (limits, per-step cap, EE box, arm distance).

  Convert: MAX_EPISODES=50 TASK_CONFIG=configs/task/dk1_pick_ee.py DATA_DIR=./data/dk1_pnp_bowl_new/success \\
           REPO_NAME=expo_ft/dk1_pnp_bowl_new_ee_50 bash scripts/dk1_task/convert_data.sh
"""

from client.envs.dk1_ee import make_ee_env
from client.envs.dk1_env import DK1Env
from configs.task import dk1_pick


def get_config():
    config = dk1_pick.get_config()
    config.env_name = "dk1_pick_ee"
    config.env = make_ee_env(DK1Env)
    config.ee_control = True
    return config
