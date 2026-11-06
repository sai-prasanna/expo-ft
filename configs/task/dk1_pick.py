"""DK1 pick-and-place: one arm puts the blue cube into the bowl. Simpler than dk1_task (stacking)
to get a reliable SFT base (target 20-50% success) for testing RL without interventions.

  Collect:  SAVE_ROOT=data/dk1_pnp_blue_bowl TASK_CONFIG=configs/task/dk1_pick.py bash scripts/dk1_task/collect_data.sh
  RL client: TASK_CONFIG=configs/task/dk1_pick_rl.py bash scripts/dk1_task/run_policy.sh

Scene: bowl at a fixed taped spot; cube placed within about +-4-5 cm of its taped spot.
Always use the same arm; the other arm stays at home.
Success (for the operator): cube inside the bowl, gripper open and clear of the bowl.
"""

from configs.task import dk1_base


def get_config():
    config = dk1_base.get_config()
    config.env_name = "dk1_pick"
    config.language_instruction = "Put the blue cube into the bowl"  # data/dk1_pnp_blue_bowl
    config.control_hz = 15
    config.auto_reset_steps = 300  # 20 s; expect demos of ~8-12 s
    return config
