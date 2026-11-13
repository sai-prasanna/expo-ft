"""DK1 action space ("joint" or "ee_cmd", expo_ft/env/dk1_ee.py), stored with the weights.

The LeRobot converter writes <dataset>/meta/action_space.json; openpi copies it into every SFT
checkpoint as <step>/assets/<asset_id>/action_space.json. The server reads it from there and EnvClientWrapper converts EE <-> joints.
"""

import glob
import json
import os


def write(directory, action_space):
    assert action_space in ("joint", "ee_cmd"), action_space
    with open(os.path.join(directory, "action_space.json"), "w") as f:
        json.dump({"action_space": action_space}, f)


def resolve(model_config, config_task):
    """Action space of the SFT checkpoint; sets config_task.ee_control to match. None for non-DK1."""
    if config_task.get("env_type") != "dk1":
        return None
    step_dir = os.path.dirname(model_config.pi05_weight_loader_path.rstrip("/"))  # .../<step>/params
    files = glob.glob(os.path.join(step_dir, "assets", "*", "*", "action_space.json"))
    if not files:
        raise ValueError(f"No assets/*/*/action_space.json in the SFT checkpoint {step_dir}")
    with open(files[0]) as f:
        space = json.load(f)["action_space"]
    if space == "ee":
        raise ValueError(f"{files[0]}: 'ee' (EE actions relative to the measured joints) was replaced by "
                         "'ee_cmd' on 2026-11-13 (relative to the last command); retrain from a converted dataset")
    assert space in ("joint", "ee_cmd"), space
    with config_task.unlocked():
        config_task.ee_control = space == "ee_cmd"
    return space
