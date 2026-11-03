"""First DK1 task. PLACEHOLDERS: set the prompt (and success notes) before collecting demos.

Success is reported manually: type 1 + Enter (success) or 2 + Enter (reset) in the
client terminal. Home pose comes from configs/task/dk1_hardware.json (`reset_joints`).
"""

from configs.task import dk1_base


def get_config():
    config = dk1_base.get_config()

    config.env_name = "dk1_task"
    # TODO(lab): the language prompt. It is baked into the LeRobot dataset as the task string,
    # so change it before converting demos.
    config.language_instruction = "PLACEHOLDER: describe the task"
    # Success (for the operator): PLACEHOLDER, e.g. "object lifted and held above the table".

    config.control_hz = 15
    config.auto_reset_steps = 300

    return config
