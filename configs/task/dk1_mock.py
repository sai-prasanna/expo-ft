"""DK1 task with fake arms, leaders and cameras: tests the whole pipeline without hardware."""

from configs.task import dk1_task


def get_config():
    config = dk1_task.get_config()
    config.env_name = "dk1_mock"
    config.language_instruction = "mock task"
    config.mock = True
    config.wait_for_scene_reset = False
    config.auto_reset_steps = 40
    # Any in-limit home pose works for the mock arms.
    import numpy as np
    config.reset_joints = np.array([0.0, 0.8, 1.2, 0.0, 0.0, 0.0, 0.0] * 2)
    config.ee_bounds = None  # fake arms: no recorded Cartesian safety
    return config
