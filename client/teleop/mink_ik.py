"""TRLC-DK1 forward/inverse kinematics with Mink + MuJoCo.

Ported from amazon-far/abc `deploy/robot/control/mink_ik.py` (commit d0832d1), with the
YAM model swapped for the DK1 follower URDF.

URDF handling (see `load_dk1_arm_model`):
- meshes are replaced by tiny spheres (MuJoCo's URDF importer mangles mesh subdirectories,
  same trick as `trlc_dk1_control/gravity_comp.py`);
- the prismatic finger joints and finger links are removed, so the model is exactly 6-DoF;
- `fusestatic="false"` keeps the fixed `tool0` frame as its own body (MuJoCo merges
  fixed-joint bodies by default, which would make `tool0` disappear).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = _REPO_ROOT / "third_party" / "trlc-dk1" / "urdf" / "follower" / "TRLC-DK1-Follower.urdf"
ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
EE_FRAME = "tool0"


def dk1_arm_xml(urdf_path: str | Path = DEFAULT_URDF) -> str:
    """URDF string for the 6-DoF DK1 arm with the `tool0` frame preserved."""
    root = ET.parse(str(urdf_path)).getroot()

    for geometry in root.iter("geometry"):
        mesh = geometry.find("mesh")
        if mesh is not None:
            geometry.remove(mesh)
            ET.SubElement(geometry, "sphere", {"radius": "0.001"})

    # Drop the finger joints/links: the IK solves the 6 arm joints only.
    drop_links = set()
    for joint in list(root.findall("joint")):
        if joint.get("type") in ("prismatic",):
            child = joint.find("child")
            if child is not None:
                drop_links.add(child.get("link"))
            root.remove(joint)
    for link in list(root.findall("link")):
        if link.get("name") in drop_links:
            root.remove(link)

    for old in root.findall("mujoco"):
        root.remove(old)
    mj = ET.SubElement(root, "mujoco")
    ET.SubElement(mj, "compiler", {"fusestatic": "false", "discardvisual": "true"})
    return ET.tostring(root, encoding="unicode")


def load_dk1_arm_model(urdf_path: str | Path = DEFAULT_URDF):
    import mujoco

    model = mujoco.MjModel.from_xml_string(dk1_arm_xml(urdf_path))
    if model.nq != 6 or model.nv != 6:
        raise RuntimeError(f"Expected a 6-DoF DK1 model, got nq={model.nq}, nv={model.nv}")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    if tuple(names) != ARM_JOINTS:
        raise RuntimeError(f"Unexpected DK1 joint order {names}; expected {ARM_JOINTS}")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_FRAME) < 0:
        raise RuntimeError(f"End-effector body {EE_FRAME!r} missing (fusestatic not honoured?)")
    return model


class DK1Kinematics:
    """Plain FK on the DK1 follower model (no mink needed). Positions in the arm base frame."""

    def __init__(self, urdf_path: str | Path = DEFAULT_URDF, ee_frame: str = EE_FRAME):
        import mujoco

        self._mujoco = mujoco
        self.model = load_dk1_arm_model(urdf_path)
        self.data = mujoco.MjData(self.model)
        self._body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_frame)
        self.lower = self.model.jnt_range[:, 0].copy()
        self.upper = self.model.jnt_range[:, 1].copy()

    def fk(self, q6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (pos[3], quat_wxyz[4]) of the end-effector frame."""
        self.data.qpos[:] = np.asarray(q6, dtype=float).reshape(6)
        self._mujoco.mj_kinematics(self.model, self.data)
        return self.data.xpos[self._body].copy(), self.data.xquat[self._body].copy()


class MinkArmIK:
    """Per-arm differential IK (abc `MinkArmController` defaults)."""

    def __init__(
        self,
        home: np.ndarray | None = None,
        *,
        urdf_path: str | Path = DEFAULT_URDF,
        target_frame: str = EE_FRAME,
        dt: float = 1.0 / 60.0,
        iterations: int = 4,
        position_cost: float = 50.0,
        orientation_cost: float = 5.0,
        gain: float = 0.6,
        max_velocity: float = 3.0,
        limit_margin: float = 0.1,
        limit_gain: float = 0.5,
        posture_cost: float = 5e-3,
        max_joint_delta: float = 0.12,
        solver: str = "daqp",
    ) -> None:
        import mink
        import mujoco

        self.mink = mink
        self.mujoco = mujoco
        self.dt = float(dt)
        self.iterations = max(1, int(iterations))
        self.max_joint_delta = float(max_joint_delta)
        self.solver = solver
        self.target_frame = target_frame
        self.model = load_dk1_arm_model(urdf_path)
        self.configuration = mink.Configuration(self.model)
        self.lower, self.upper = self._joint_limits(limit_margin)
        mid = 0.5 * (self.lower + self.upper)
        self.home = mid if home is None else np.clip(np.asarray(home, float).reshape(6), self.lower, self.upper)
        self.configuration.update(q=self.home)
        self.frame_task = mink.FrameTask(
            frame_name=target_frame,
            frame_type="body",
            position_cost=float(position_cost),
            orientation_cost=float(orientation_cost),
            gain=float(gain),
            lm_damping=1e-6,
        )
        posture = mink.PostureTask(self.model, cost=float(posture_cost))
        posture.set_target(mid)
        self.tasks = [self.frame_task, posture]
        self.limits = [
            mink.ConfigurationLimit(
                self.model,
                gain=float(limit_gain),
                min_distance_from_limits=self._effective_margin(limit_margin),
            )
        ]
        if max_velocity > 0:
            self.limits.append(mink.VelocityLimit(self.model, {j: float(max_velocity) for j in ARM_JOINTS}))

    def _effective_margin(self, requested: float) -> float:
        margin = max(0.0, float(requested))
        for i in range(self.model.njnt):
            if self.model.jnt_limited[i]:
                low, high = self.model.jnt_range[i]
                margin = min(margin, 0.49 * float(high - low))
        return margin

    def _joint_limits(self, requested_margin: float) -> tuple[np.ndarray, np.ndarray]:
        margin = self._effective_margin(requested_margin)
        lower, upper = np.full(6, -np.inf), np.full(6, np.inf)
        for i in range(self.model.njnt):
            if not self.model.jnt_limited[i]:
                continue
            adr = self.model.jnt_qposadr[i]
            low, high = self.model.jnt_range[i]
            if high - low > 2 * margin:
                low, high = low + margin, high - margin
            lower[adr], upper[adr] = low, high
        return lower, upper

    def fk(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(pos, quat_wxyz) of the target frame. Joints are NOT clipped (FK of the real pose)."""
        self.configuration.update(q=np.asarray(joints, dtype=float).reshape(6))
        pose = self.configuration.get_transform_frame_to_world(self.target_frame, "body").parameters()
        return np.asarray(pose[4:], dtype=float), np.asarray(pose[:4], dtype=float)

    def solve(self, target_pos: np.ndarray, target_wxyz: np.ndarray, seed: np.ndarray) -> np.ndarray:
        seed = np.clip(np.asarray(seed, dtype=float).reshape(6), self.lower, self.upper)
        self.configuration.update(q=seed)
        target = self.mink.SE3.from_rotation_and_translation(
            self.mink.SO3(np.asarray(target_wxyz, dtype=float).reshape(4)),
            np.asarray(target_pos, dtype=float).reshape(3),
        )
        self.frame_task.set_target(target)
        step_dt = self.dt / self.iterations
        try:
            for _ in range(self.iterations):
                vel = self.mink.solve_ik(
                    self.configuration, self.tasks, step_dt, self.solver, damping=1e-6, limits=self.limits
                )
                self.configuration.integrate_inplace(vel, step_dt)
        except self.mink.NoSolutionFound:
            return seed.copy()
        result = np.clip(np.asarray(self.configuration.q), self.lower, self.upper)
        result = seed + np.clip(result - seed, -self.max_joint_delta, self.max_joint_delta)
        return result if np.isfinite(result).all() else seed.copy()
