"""dk1_task with a tight per-step joint cap, for first real-robot rollouts of a new checkpoint.

0.1 rad/step at 15 Hz = 1.5 rad/s per joint (the base config's 0.25 rad/step = 3.75 rad/s barely
limits anything). Demo consecutive-action steps peak at ~0.13-0.19 rad, so this only slows the
fastest motions. Use on the robot client only:
  TASK_CONFIG=configs/task/dk1_task_safe.py bash scripts/dk1_task/run_policy.sh
"""

from configs.task import dk1_task


def get_config():
    config = dk1_task.get_config()
    config.max_joint_delta_per_step = 0.1
    return config
