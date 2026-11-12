"""dk1_pick evaluation: dk1_pick_rl.py with the eval layout stream (same E1, E2, ... every run).

  TASK_CONFIG=configs/task/dk1_pick_eval.py bash scripts/dk1_task/run_policy.sh
"""

from configs.task import dk1_pick_rl


def get_config():
    config = dk1_pick_rl.get_config()
    config.layout_guide.mode = "eval"
    return config
