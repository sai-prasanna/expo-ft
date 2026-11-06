"""dk1_task for EXPO-FT RL rollouts without human interventions (robot client only).

  TASK_CONFIG=configs/task/dk1_task_rl.py bash scripts/dk1_task/run_policy.sh

- No leader arms / clutch: interventions are off. The operator still resets the scene (Enter)
  and labels each episode (1 = success, 2 = failure).
- Joint cap 0.2 rad/step (as dk1_task_safe.py).
- EE box per arm (arm base frame, metres): the range of the end-effector over all 20
  stack_green_blue demos (FK of action14), plus 5 cm in x/y and on top, with the floor at
  z = 0.005 (demos went down to 0.017). An arm whose command would leave its box holds its
  joints; motion back toward the box is allowed. Home (FK [0.25, 0, 0.16]) is inside both.
  Demo ranges: left x 0.21-0.47 y -0.32-0.16 z 0.02-0.30; right x 0.17-0.52 y -0.06-0.53 z 0.02-0.37.
"""

from configs.task import dk1_task


def get_config():
    config = dk1_task.get_config()
    config.max_joint_delta_per_step = 0.2
    config.intervention.enabled = False
    config.ee_bounds = {
        "left": [[0.16, 0.52], [-0.37, 0.21], [0.005, 0.35]],
        "right": [[0.12, 0.57], [-0.11, 0.58], [0.005, 0.42]],
    }
    return config
