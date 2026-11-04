"""Bimanual TRLC-DK1 base task config (EXPO-FT port). Tasks inherit from this (see dk1_task.py).

Hardware specifics (serial ports, cameras, home pose) are machine-specific and live in
`configs/task/dk1_hardware.json`, written by

    python -m client.dk1_setup_hardware            # ports + cameras (interactive)

If that file is missing, the PLACEHOLDER values below are used (fine for mock=True; a real
DK1Env refuses to start with placeholder ports).

Every key here is passed to DK1Env(**config) by run_client / collect_data_dk1, so DK1Env
accepts **kwargs.
"""

import json
from pathlib import Path

import ml_collections
import numpy as np

try:
    from client.envs.dk1_env import DK1Env
except Exception as e:  # server venv without client deps, etc.
    DK1Env = None
    print(f"Not importing DK1 env ({e!r})")

HARDWARE_FILE = Path(__file__).with_name("dk1_hardware.json")
# Folded-up rest pose = calibration zero: all joints 0, grippers open. Used as the default home.
DK1_FOLDED_POSE = [0.0] * 14


def _load_hardware() -> dict:
    if HARDWARE_FILE.exists():
        with open(HARDWARE_FILE) as f:
            return json.load(f)
    return {}


def get_config():
    config = ml_collections.ConfigDict()

    config.env_type = "dk1"
    config.env = DK1Env
    config.env_name = "dk1"
    config.language_instruction = "PLACEHOLDER: set in the task config"

    hw = _load_hardware()

    # ---- hardware (overridden by dk1_hardware.json) -------------------------------------
    config.left_follower_port = hw.get("left_follower_port", "PLACEHOLDER_LEFT_FOLLOWER_PORT")
    config.right_follower_port = hw.get("right_follower_port", "PLACEHOLDER_RIGHT_FOLLOWER_PORT")
    config.left_leader_port = hw.get("left_leader_port", "PLACEHOLDER_LEFT_LEADER_PORT")
    config.right_leader_port = hw.get("right_leader_port", "PLACEHOLDER_RIGHT_LEADER_PORT")
    # name -> LeRobot OpenCVCameraConfig kwargs. Names must be head / left_wrist / right_wrist.
    config.cameras = hw.get("cameras", {
        "head": {"index_or_path": "PLACEHOLDER_HEAD_CAMERA", "width": 640, "height": 480, "fps": 30},
        "left_wrist": {"index_or_path": "PLACEHOLDER_LEFT_WRIST_CAMERA", "width": 640, "height": 480, "fps": 30},
        "right_wrist": {"index_or_path": "PLACEHOLDER_RIGHT_WRIST_CAMERA", "width": 640, "height": 480, "fps": 30},
    })
    config.control_mode = "impedance"  # DK1Follower: "impedance" (MIT + gravity comp) or "pos_vel"
    config.max_gripper_torque = 1.0  # Nm
    config.use_bimanual_class = False  # True: BiDK1Follower (its feature listing references a missing `.motors`)
    config.leader_gripper_open_pos = 2280
    config.leader_gripper_closed_pos = 1670

    # Home pose, 14-D [L j1..j6, L grip, R j1..j6, R grip]. Default: the DK1 folded pose, which
    # is the calibration zero (all joints 0, grippers open). To use another pose, add a 14-value
    # "reset_joints" list to dk1_hardware.json.
    reset = hw.get("reset_joints", DK1_FOLDED_POSE)
    config.reset_joints = np.asarray(reset, dtype=np.float64)
    config.reset_max_joint_speed = 0.3  # rad/s while homing (~17 deg/s for the joint with the farthest to go)
    config.wait_for_scene_reset = True  # press Enter after the manual scene reset

    # ---- observation / control ----------------------------------------------------------
    config.image_size = (240, 320)  # 4:3 to match the 640x480 cameras (no padding); the model pads to 224x224
    config.control_hz = 15
    config.auto_reset_steps = 300  # 20 s at 15 Hz
    config.ignore_auto_reset = False

    # Safety: per-step joint change limit (rad) w.r.t. the measured joints, used for BOTH demo
    # collection and policy rollouts (demos store the clipped action, so the policy never learns
    # moves larger than RL allows). It bounds how far the command leads the measured pose. 0.25 rad
    # (~14 deg) felt right in teleop with smoothing on; 0.1-0.2 felt held back. Lower it if policy
    # rollouts are too fast.
    config.max_joint_delta_per_step = 0.25
    # Optional EE box per arm in that arm's base frame (m), [[xmin,xmax],[ymin,ymax],[zmin,zmax]].
    config.ee_bounds = None
    # On: 15 Hz steps caused micro jerks in teleop. A 100 Hz thread
    # ramps each new target over one control period (adds ~66 ms lag). 0.0 = off.
    config.command_interp_hz = 100.0
    # Pose the arms move to before the motors are disabled on close (they go limp there).
    # Defaults to the home pose; set "rest_joints" in dk1_hardware.json to use another pose.
    rest = hw.get("rest_joints", reset)
    config.rest_joints = None if rest is None else np.asarray(rest, dtype=np.float64)

    # ---- EXPO-FT plumbing -----------------------------------------------------------------
    config.action_space = "action14"  # traj.hdf5 action/action14 holds [L6, Lg, R6, Rg]
    config.gripper_action_space = None  # gripper already inside action14
    # Server sends this at startup; all -1 = "hold" (zeros would command every joint to 0 rad).
    config.example_action = -np.ones((1, 14))
    config.edit_action_xyzg = False  # DROID-Cartesian-only option
    config.critic_camera_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

    # ---- interventions: anchored relative IK from the leader arms ------------------------
    config.intervention = ml_collections.ConfigDict({
        "enabled": True,
        "clutch_backend": "terminal",  # "terminal" (second terminal, UDP) | "pynput" (X11) | "manual"
        "clutch_port": 8765,
        "clutch_key": "f8",  # pynput only; never 1/2 (manual success detector)
        "rate_hz": 60.0,
        "position_scale": 1.0,
        "rotation_scale": 1.0,
        # leader -> follower joint map, q_f = sign * q_l + offset, [L j1..j6, R j1..j6].
        # Check on hardware with `python -m client.teleop.test_relative_ik --hardware`.
        "leader_joint_offsets": [0.0] * 12,
        "leader_joint_signs": [1.0] * 12,
        # mink IK (abc defaults)
        "iterations": 4,
        "position_cost": 50.0,
        "orientation_cost": 5.0,
        "max_velocity": 3.0,
        "max_joint_delta": 0.12,
        "solver": "daqp",
    })

    config.mock = False
    return config
