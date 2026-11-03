"""Bimanual TRLC-DK1 environment for the EXPO-FT client (duck-typed like `DroidEnv`).

Action and state: 14-D absolute joint positions
    [L joint_1..joint_6, L gripper, R joint_1..joint_6, R gripper]
joints in radians, grippers 0 (open) .. 1 (closed), the DK1 LeRobot plugin's convention.

Observation (flat dict; keys == LeRobot feature names == openpi repack keys):
    head_image, left_wrist_image, right_wrist_image : uint8 HxWx3 RGB, resized to `image_size`
    state                                           : float32 (14,)
    prompt                                          : str

An all -1 action means "hold the current position" (the server sends it while it has no
plan, e.g. at startup and after a human hand-back).

`mock=True` replaces the arms, leaders and cameras with fakes so the whole pipeline
(collection, conversion, server handshake, interventions) runs without hardware.

Nothing here imports `droid`; LeRobot / trlc-dk1 are imported lazily (only for real hardware),
so the server venv can import this module (via configs/task/dk1_*.py) without them.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

from client.envs.utils import process_image_for_obs
from client.real_utils.detector import success_detector_manual

ARM_JOINT_KEYS = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
SIDES = ("left", "right")
IMAGE_KEYS = {"head": "head_image", "left_wrist": "left_wrist_image", "right_wrist": "right_wrist_image"}

# trlc_dk1_control/config.py DK1RobotConfig.joint_pos_limits (kept in sync with the URDF).
DK1_JOINT_LIMITS = np.array([
    [-2.0943951023931953, 2.0943951023931953],
    [-0.08726646259971647, 3.141592653589793],
    [-0.08726646259971647, 4.71238898038469],
    [-1.95, 1.5707963267948966],
    [-1.5707963267948966, 1.5707963267948966],
    [-2.0943951023931953, 2.0943951023931953],
])
# DK1Robot never commands within LIMIT_BUFFER of a hard limit; stay inside it too.
LIMIT_BUFFER = 0.05


def _is_hold(action: np.ndarray) -> bool:
    return bool(np.allclose(action, -1.0))


def arm_slice(side: str) -> slice:
    return slice(0, 7) if side == "left" else slice(7, 14)


def clip_to_limits(action14: np.ndarray) -> np.ndarray:
    a = np.asarray(action14, dtype=np.float64).reshape(14).copy()
    lo, hi = DK1_JOINT_LIMITS[:, 0] + LIMIT_BUFFER, DK1_JOINT_LIMITS[:, 1] - LIMIT_BUFFER
    for side in SIDES:
        arm = a[arm_slice(side)]
        arm[:6] = np.clip(arm[:6], lo, hi)
        arm[6] = np.clip(arm[6], 0.0, 1.0)
    return a


def clip_joint_delta(action14: np.ndarray, current14: np.ndarray, max_delta: float | None) -> np.ndarray:
    """Limit each arm joint's change w.r.t. the measured state; grippers are not limited."""
    if max_delta is None or max_delta <= 0:
        return action14
    a = action14.copy()
    for side in SIDES:
        sl = arm_slice(side)
        cur = current14[sl][:6]
        a[sl.start:sl.start + 6] = cur + np.clip(a[sl.start:sl.start + 6] - cur, -max_delta, max_delta)
    return a


# ====================================================================== hardware backends
class _MockArms:
    """Two fake arms that track the commanded joints with a first-order lag."""

    def __init__(self, init14: np.ndarray, control_hz: float):
        self.q = np.asarray(init14, dtype=np.float64).copy()
        self.target = self.q.copy()
        self._t = time.perf_counter()
        self._tau = 0.5 / max(control_hz, 1.0)  # ~ half a control step

    def _advance(self):
        now = time.perf_counter()
        alpha = 1.0 - np.exp(-(now - self._t) / self._tau)
        self._t = now
        self.q += alpha * (self.target - self.q)

    def get(self) -> np.ndarray:
        self._advance()
        return self.q.copy()

    def send(self, action14: np.ndarray, interpolate: bool = True):
        self._advance()
        self.target = np.asarray(action14, dtype=np.float64).copy()

    def close(self):
        pass


class _RealArms:
    """Two DK1 followers (or one BiDK1Follower) through the LeRobot plugin.

    Thread-safe (one lock around all serial access; the IK thread reads the followers while the
    main thread commands them). With interp_hz > 0, a background thread ramps each new target
    linearly over `interp_s` (one control period) instead of letting the 250 Hz impedance loop
    see a step of up to max_joint_delta_per_step every 1/control_hz.
    """

    def __init__(self, left_port, right_port, control_mode="impedance", max_gripper_torque=1.0,
                 use_bimanual_class=False, interp_hz: float = 0.0, interp_s: float = 0.0):
        from lerobot_robot_trlc_dk1 import BiDK1Follower, BiDK1FollowerConfig, DK1Follower, DK1FollowerConfig

        self._lock = threading.RLock()
        self.bimanual = bool(use_bimanual_class)
        if self.bimanual:
            # NOTE: BiDK1FollowerConfig does not forward control_mode (always "impedance"), and its
            # observation_features/action_features reference a missing `.motors`; get/send work.
            self.robot = BiDK1Follower(BiDK1FollowerConfig(
                left_arm_port=left_port, right_arm_port=right_port, max_gripper_torque=max_gripper_torque))
            self.robot.connect()
        else:
            self.arms = {}
            for side, port in (("left", left_port), ("right", right_port)):
                arm = DK1Follower(DK1FollowerConfig(
                    port=port, control_mode=control_mode, max_gripper_torque=max_gripper_torque))
                arm.connect()
                self.arms[side] = arm
        self._interp_hz = float(interp_hz)
        self._interp_s = float(interp_s)
        self._seg = None  # (start14, end14, t0)
        self._stop = threading.Event()
        self._thread = None
        if self._interp_hz > 0 and self._interp_s > 0:
            self._thread = threading.Thread(target=self._interp_loop, daemon=True, name="dk1-interp")
            self._thread.start()

    def get(self) -> np.ndarray:
        with self._lock:
            if self.bimanual:
                obs = self.robot.get_observation()
                return np.array([obs[f"{s}_{k}"] for s in SIDES for k in ARM_JOINT_KEYS], dtype=np.float64)
            out = []
            for side in SIDES:
                obs = self.arms[side].get_observation()
                out.extend(obs[k] for k in ARM_JOINT_KEYS)
            return np.asarray(out, dtype=np.float64)

    def _send_now(self, a: np.ndarray):
        with self._lock:
            if self.bimanual:
                self.robot.send_action({f"{s}_{k}": float(a[arm_slice(s)][i])
                                        for s in SIDES for i, k in enumerate(ARM_JOINT_KEYS)})
                return
            for side in SIDES:
                arm = a[arm_slice(side)]
                self.arms[side].send_action({k: float(arm[i]) for i, k in enumerate(ARM_JOINT_KEYS)})

    def _current_command(self, now: float):
        seg = self._seg
        if seg is None:
            return None
        start, end, t0 = seg
        alpha = min(1.0, (now - t0) / self._interp_s)
        return start + (end - start) * alpha

    def send(self, action14: np.ndarray, interpolate: bool = True):
        a = np.asarray(action14, dtype=np.float64).copy()
        if self._thread is None or not interpolate:
            self._seg = (a, a, time.perf_counter())
            self._send_now(a)
            return
        now = time.perf_counter()
        cur = self._current_command(now)
        start = a if cur is None else cur
        self._seg = (start, a, now)

    def _interp_loop(self):
        period = 1.0 / self._interp_hz
        while not self._stop.is_set():
            t0 = time.perf_counter()
            cmd = self._current_command(t0)
            if cmd is not None:
                try:
                    self._send_now(cmd)
                except Exception as e:
                    print(f"[DK1Env] interpolation send error: {e!r}")
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        robots = [self.robot] if self.bimanual else list(self.arms.values())
        for r in robots:
            try:
                r.disconnect()
            except Exception as e:
                print(f"[DK1Env] disconnect error: {e!r}")


class _MockLeaders:
    """Fake leaders: follow a settable 14-D pose (tests drive it via `pose`)."""

    def __init__(self, init14: np.ndarray):
        self.pose = np.asarray(init14, dtype=np.float64).copy()

    def get(self) -> np.ndarray:
        return self.pose.copy()

    def close(self):
        pass


class _RealLeaders:
    def __init__(self, left_port, right_port, gripper_open_pos=2280, gripper_closed_pos=1670):
        from lerobot_robot_trlc_dk1 import DK1Leader, DK1LeaderConfig

        self.arms = {}
        for side, port in (("left", left_port), ("right", right_port)):
            leader = DK1Leader(DK1LeaderConfig(
                port=port, gripper_open_pos=gripper_open_pos, gripper_closed_pos=gripper_closed_pos))
            leader.connect()
            self.arms[side] = leader

    def get(self) -> np.ndarray:
        out = []
        for side in SIDES:
            act = self.arms[side].get_action()
            out.extend(act[k] for k in ARM_JOINT_KEYS)
        a = np.asarray(out, dtype=np.float64)
        a[6], a[13] = np.clip(a[6], 0.0, 1.0), np.clip(a[13], 0.0, 1.0)
        return a

    def close(self):
        for leader in self.arms.values():
            try:
                leader.disconnect()
            except Exception as e:
                print(f"[DK1Env] leader disconnect error: {e!r}")


class _MockCameras:
    def __init__(self, names, height=480, width=640):
        self.names = list(names)
        self.shape = (height, width, 3)
        self._rng = np.random.default_rng(0)

    def read(self) -> dict:
        # Values in [2, 255] so Pi05.process_raw_inputs' `max > 1` image assertion holds.
        return {n: self._rng.integers(2, 256, size=self.shape, dtype=np.uint8) for n in self.names}

    def close(self):
        pass


class _RealCameras:
    def __init__(self, cameras: dict):
        from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

        self.cams = {}
        for name, spec in cameras.items():
            spec = dict(spec)
            iop = spec.pop("index_or_path")
            if isinstance(iop, str) and iop.isdigit():
                iop = int(iop)
            elif isinstance(iop, str):
                # Resolve /dev/v4l/by-id|by-path symlinks to /dev/videoN for OpenCV's V4L2 backend.
                from pathlib import Path
                iop = Path(os.path.realpath(iop))
            cfg = OpenCVCameraConfig(index_or_path=iop, fps=spec.pop("fps", 30),
                                     width=spec.pop("width", 640), height=spec.pop("height", 480), **spec)
            cam = OpenCVCamera(cfg)
            cam.connect()
            self.cams[name] = cam

    def read(self) -> dict:
        return {name: cam.async_read() for name, cam in self.cams.items()}  # RGB uint8 HxWx3

    def close(self):
        for cam in self.cams.values():
            try:
                cam.disconnect()
            except Exception:
                pass


# ====================================================================== env
class DK1Env:
    def __init__(
        self,
        *,
        left_follower_port: str = "",
        right_follower_port: str = "",
        left_leader_port: str = "",
        right_leader_port: str = "",
        cameras: dict | None = None,
        control_mode: str = "impedance",
        max_gripper_torque: float = 1.0,
        use_bimanual_class: bool = False,
        leader_gripper_open_pos: int = 2280,
        leader_gripper_closed_pos: int = 1670,
        image_size=(180, 320),
        control_hz: float = 15,
        language_instruction: str = "",
        reset_joints=None,
        reset_max_joint_speed: float = 0.5,
        reset_settle_s: float = 0.5,
        wait_for_scene_reset: bool = True,
        auto_reset_steps: int = 300,
        ignore_auto_reset: bool = False,
        max_joint_delta_per_step: float | None = 0.1,
        ee_bounds=None,
        command_interp_hz: float = 0.0,
        rest_joints=None,
        intervention: dict | None = None,
        mock: bool = False,
        video_dir: str | None = None,
        env_usage: str = "train",
        **kwargs,  # the whole task config is passed in; ignore unrelated keys
    ):
        self.mock = bool(mock)
        self.control_hz = float(control_hz)
        self.image_size = tuple(image_size) if image_size is not None else None
        self.language_instruction = language_instruction
        self.reset_joints = None if reset_joints is None else np.asarray(reset_joints, dtype=np.float64).reshape(14)
        self.reset_max_joint_speed = float(reset_max_joint_speed)
        self.reset_settle_s = float(reset_settle_s)
        self.wait_for_scene_reset = bool(wait_for_scene_reset) and not self.mock
        self.auto_reset_steps = int(auto_reset_steps)
        self.ignore_auto_reset = bool(ignore_auto_reset)
        self.max_joint_delta_per_step = max_joint_delta_per_step
        self.ee_bounds = None if ee_bounds is None else {s: np.asarray(ee_bounds[s], float).reshape(3, 2) for s in SIDES}
        self.intervention_cfg = dict(intervention or {})
        self.rest_joints = None if rest_joints is None else np.asarray(rest_joints, dtype=np.float64).reshape(14)
        self._closed = False
        self.env_usage = env_usage
        self.video_dir = video_dir
        self._steps_since_reset = 0
        self._clip_count = 0
        self._last_command = None
        self._intervention = None
        self._clutch = None
        self._leaders = None
        self._kin = None
        self.prev_obs = None
        self.done, self.success = False, False

        cameras = dict(cameras or {})
        self.camera_names = list(cameras) or list(IMAGE_KEYS)
        missing = set(IMAGE_KEYS) - set(self.camera_names)
        if missing:
            raise ValueError(f"cameras must define {sorted(IMAGE_KEYS)}; missing {sorted(missing)}")

        self._leader_kwargs = dict(left_port=left_leader_port, right_port=right_leader_port,
                                   gripper_open_pos=leader_gripper_open_pos,
                                   gripper_closed_pos=leader_gripper_closed_pos)
        if self.mock:
            init = self.reset_joints if self.reset_joints is not None else np.zeros(14)
            self._arms = _MockArms(init, self.control_hz)
            self._cams = _MockCameras(self.camera_names)
        else:
            for name, port in (("left_follower_port", left_follower_port), ("right_follower_port", right_follower_port)):
                if not port or "PLACEHOLDER" in str(port):
                    raise ValueError(f"{name} is not set; run `python -m client.dk1_setup_hardware` first.")
            if self.reset_joints is None and env_usage in ("train", "eval"):
                raise ValueError("reset_joints is not set: RL/eval episodes must start from a home pose. "
                                 "Run `python -m client.dk1_setup_hardware --record_home`.")
            self._arms = _RealArms(left_follower_port, right_follower_port, control_mode=control_mode,
                                   max_gripper_torque=max_gripper_torque, use_bimanual_class=use_bimanual_class,
                                   interp_hz=command_interp_hz, interp_s=1.0 / self.control_hz)
            self._cams = _RealCameras(cameras)
        # RL rollouts: connect leaders + clutch + IK now, so a broken intervention setup fails
        # loudly at create_env instead of silently leaving the operator without a takeover.
        if env_usage == "train" and self.intervention_cfg.get("enabled", True) and not self.mock:
            self._ensure_intervention()

    # ------------------------------------------------------------------ observation
    def get_state(self) -> np.ndarray:
        return self._arms.get()

    def get_raw_observation(self) -> dict:
        raw = {"state": self._arms.get(), "images": self._cams.read(), "timestamp": time.time()}
        self.prev_obs = raw
        return raw

    def transform_observation(self, raw: dict) -> dict:
        obs = {}
        for cam, key in IMAGE_KEYS.items():
            obs[key] = process_image_for_obs(np.asarray(raw["images"][cam]), bgr_to_rgb=False, image_size=self.image_size)
        obs["state"] = np.asarray(raw["state"], dtype=np.float32)
        obs["prompt"] = self.language_instruction
        return obs

    def get_observation(self) -> dict:
        return self.transform_observation(self.get_raw_observation())

    def render(self):
        return np.asarray(self._cams.read()["head"])

    # ------------------------------------------------------------------ action
    def _kinematics(self):
        if self._kin is None:
            from client.teleop.mink_ik import DK1Kinematics
            self._kin = DK1Kinematics()
        return self._kin

    def _enforce_ee_bounds(self, action14: np.ndarray, current14: np.ndarray) -> np.ndarray:
        """If an arm's commanded EE position leaves its box, hold that arm's joints (gripper still moves)."""
        if self.ee_bounds is None:
            return action14
        kin = self._kinematics()
        a = action14.copy()
        for side in SIDES:
            sl = arm_slice(side)
            pos, _ = kin.fk(a[sl][:6])
            box = self.ee_bounds[side]
            if np.any(pos < box[:, 0]) or np.any(pos > box[:, 1]):
                cur_pos, _ = kin.fk(current14[sl][:6])
                # allow motion that moves back toward the box
                cur_out = np.maximum(box[:, 0] - cur_pos, 0) + np.maximum(cur_pos - box[:, 1], 0)
                new_out = np.maximum(box[:, 0] - pos, 0) + np.maximum(pos - box[:, 1], 0)
                if new_out.sum() >= cur_out.sum():
                    a[sl.start:sl.start + 6] = current14[sl][:6]
        return a

    def safe_action(self, action, max_delta: float | None = None) -> np.ndarray:
        """Clip to joint limits, per-step joint change and the optional EE box. Returns 14-D."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)[:14]
        current = self._arms.get()
        if _is_hold(action):
            hold = self._last_command if self._last_command is not None else current
            return hold.copy()
        a = clip_to_limits(action)
        before = a.copy()
        a = clip_joint_delta(a, current, self.max_joint_delta_per_step if max_delta is None else max_delta)
        self._clip_count += int(not np.allclose(a, before))
        a = self._enforce_ee_bounds(a, current)
        return a

    def step(self, action, max_delta: float | None = None) -> dict:
        self._steps_since_reset += 1
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] < 14:
            raise ValueError(f"DK1Env expects a 14-D action, got {action.shape}")
        executed = self.safe_action(action, max_delta=max_delta)
        self._arms.send(executed)
        self._last_command = executed.copy()
        return {"executed_action": executed}

    # ------------------------------------------------------------------ reset / termination
    def _move_to(self, target14: np.ndarray):
        """Blocking linear interpolation to target14 at <= reset_max_joint_speed (grippers included)."""
        start = self._arms.get()
        dist = np.max(np.abs(target14 - start)[np.r_[0:6, 7:13]])
        duration = max(dist / max(self.reset_max_joint_speed, 1e-3), 0.2)
        rate = 50.0
        n = int(np.ceil(duration * rate))
        for i in range(1, n + 1):
            t0 = time.perf_counter()
            cmd = start + (target14 - start) * (i / n)
            cmd = clip_to_limits(cmd)
            self._arms.send(cmd, interpolate=False)
            self._last_command = cmd
            if not self.mock:
                time.sleep(max(0.0, 1.0 / rate - (time.perf_counter() - t0)))
        if not self.mock:
            time.sleep(self.reset_settle_s)

    def _wait_for_enter(self, msg: str):
        print(msg, flush=True)
        try:
            with open("/dev/tty", "r") as tty:
                tty.readline()
        except OSError:
            sys.stdin.readline()

    def reset(self) -> dict:
        self._release_intervention()
        if self._steps_since_reset > 0:
            print(f"[DK1Env] per-step joint cap hit on {self._clip_count}/{self._steps_since_reset} steps "
                  f"({100.0 * self._clip_count / self._steps_since_reset:.0f}%) last episode")
        self._clip_count = 0
        if self.reset_joints is not None:
            target = self.reset_joints.copy()
            target[6] = target[13] = 0.0  # open grippers
            self._move_to(target)
        if self.wait_for_scene_reset:
            self._wait_for_enter("[DK1Env] Reset the scene, then press Enter to start the episode.")
        self._steps_since_reset = 0
        self.done, self.success = False, False
        return self.get_observation()

    def detect(self, raw_obs):
        """Override for an automatic success detector: returns (success, terminate)."""
        return False, False

    def auto_reset_due(self) -> bool:
        if self.ignore_auto_reset:
            return False
        due = self._steps_since_reset >= self.auto_reset_steps
        if due:
            print("Timeout...")
        return due

    def get_info_for_step(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.prev_obs
        time_stop = self.auto_reset_due()
        manual = success_detector_manual()
        if manual == "success":
            done, success, manual_stop = True, True, True
        elif manual == "reset":
            done, success, manual_stop = True, False, True
        else:
            manual_stop = False
            success, terminate = self.detect(raw_obs)
            done = bool(success or terminate or time_stop)
        if done:
            print(f"Done! Success: {success}, Time stop: {time_stop}, Manual stop: {manual_stop}")
        self.done, self.success = done, success
        reward = 1.0 if success else 0.0
        mask = 0.0 if done else 1.0
        return done, success, reward, mask

    @property
    def steps_since_reset(self):
        return self._steps_since_reset

    # ------------------------------------------------------------------ leaders / interventions
    def get_leaders(self):
        """Lazily connect the leader arms (or the mock leaders)."""
        if self._leaders is None:
            if self.mock:
                self._leaders = _MockLeaders(self._arms.get())
            else:
                for side in SIDES:
                    port = self._leader_kwargs[f"{side}_port"]
                    if not port or "PLACEHOLDER" in str(port):
                        raise ValueError(f"{side}_leader_port is not set; run `python -m client.dk1_setup_hardware`.")
                self._leaders = _RealLeaders(**self._leader_kwargs)
        return self._leaders

    def get_clutch(self):
        if self._clutch is None:
            from client.teleop.clutch import Clutch
            cfg = self.intervention_cfg
            backend = cfg.get("clutch_backend", "terminal")
            self._clutch = Clutch(backend=("manual" if self.mock and backend == "terminal" and cfg.get("mock_manual_clutch", True) else backend),
                                  port=cfg.get("clutch_port", 8765), key=cfg.get("clutch_key", "f8"))
        return self._clutch

    def _ensure_intervention(self):
        if self._intervention is None:
            from client.teleop.relative_ik import InterventionThread, RelativeIKController
            cfg = self.intervention_cfg
            ik_keys = ("iterations", "position_cost", "orientation_cost", "gain", "max_velocity",
                       "limit_margin", "limit_gain", "posture_cost", "max_joint_delta", "solver")
            ik_kwargs = {k: cfg[k] for k in ik_keys if k in cfg}
            rate = float(cfg.get("rate_hz", 60.0))
            controller = RelativeIKController(
                home14=self.reset_joints,
                position_scale=cfg.get("position_scale", 1.0),
                rotation_scale=cfg.get("rotation_scale", 1.0),
                leader_joint_offsets=cfg.get("leader_joint_offsets"),
                leader_joint_signs=cfg.get("leader_joint_signs"),
                dt=1.0 / rate,
                **ik_kwargs,
            )
            leaders = self.get_leaders()
            self._intervention = InterventionThread(controller, self.get_clutch(), leaders.get, self._arms.get, rate_hz=rate)
        return self._intervention

    def get_human_action(self):
        """(action14 or None, is_human). Starts the leaders/clutch/IK thread on first call."""
        if not self.intervention_cfg.get("enabled", True):
            return None, False
        try:
            thread = self._ensure_intervention()
        except Exception as e:
            if not self.mock:
                raise RuntimeError(f"DK1 intervention setup failed: {e!r}. Fix it or set "
                                   f"intervention.enabled=False in the task config.") from e
            print(f"[DK1Env] intervention unavailable ({e!r}); policy keeps control.")
            self.intervention_cfg["enabled"] = False
            return None, False
        return thread.get()

    def _release_intervention(self):
        if self._clutch is not None and self._clutch.engaged:
            self._clutch.set_engaged(False)

    def park(self):
        """Before the motors are disabled (DK1Follower.disconnect() makes the arms go limp): move to
        rest_joints, or to the home pose if none is set. Without an explicit rest_joints (a pose
        that is safe to go limp in), also ask the operator to support the arms."""
        if self.mock:
            return
        try:
            target = self.rest_joints if self.rest_joints is not None else self.reset_joints
            if target is not None:
                print("[DK1Env] moving to " + ("rest_joints" if self.rest_joints is not None else "the home pose")
                      + " before disabling the motors...")
                self._move_to(target)
            if self.rest_joints is None:
                self._wait_for_enter("[DK1Env] !!! Motors are about to be DISABLED and the arms will fall. "
                                     "Support both arms, then press Enter.")
        except BaseException as e:  # Ctrl-C during park: still disconnect cleanly
            print(f"[DK1Env] park interrupted ({e!r})")

    def close(self, park: bool = True):
        if self._closed:
            return
        self._closed = True
        if park:
            self.park()
        for obj in (self._intervention, self._clutch, self._leaders, self._cams, self._arms):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        self._intervention = self._clutch = self._leaders = None


__all__ = ["DK1Env", "clip_to_limits", "clip_joint_delta", "IMAGE_KEYS"]
