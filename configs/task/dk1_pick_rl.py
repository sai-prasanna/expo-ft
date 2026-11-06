"""dk1_pick for EXPO-FT RL rollouts without interventions (same safety as dk1_task_rl.py)."""

from configs.task import dk1_pick


def get_config():
    config = dk1_pick.get_config()
    config.max_joint_delta_per_step = 0.2
    config.max_command_delta_per_step = 0.1
    config.intervention.enabled = False
    return config
