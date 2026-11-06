"""dk1_task for EXPO-FT RL rollouts without human interventions (robot client only).

  TASK_CONFIG=configs/task/dk1_task_rl.py bash scripts/dk1_task/run_policy.sh

- No leader arms / clutch: interventions are off. The operator still resets the scene (Enter)
  and labels each episode (1 = success, 2 = failure).
- Joint cap 0.2 rad/step (as dk1_task_safe.py).
- Cartesian safety (EE box per arm + 8 cm arm-to-arm distance) from dk1_hardware.json, recorded
  with `python -m client.dk1_record_safety`. If no box is recorded yet, a demo-derived one is used:
  the tool0 range over all 20 stack_green_blue demos plus 5 cm in x/y and on top, floor at
  z = 0.005 (demos went down to 0.017). Home (FK [0.25, 0, 0.16]) is inside both.
"""

from configs.task import dk1_task


def get_config():
    config = dk1_task.get_config()
    config.max_joint_delta_per_step = 0.2
    config.intervention.enabled = False
    if config.ee_bounds is None:
        config.ee_bounds = {
            "left": [[0.16, 0.52], [-0.37, 0.21], [0.005, 0.35]],
            "right": [[0.12, 0.57], [-0.11, 0.58], [0.005, 0.42]],
        }
    return config
