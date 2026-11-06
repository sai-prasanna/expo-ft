"""Cartesian safety for the bimanual DK1: an EE box per arm and a minimum arm-to-arm distance.

Both act on the joint command after the per-step joint cap (DK1Env.safe_action), with the same rule:
a command that makes a violation worse holds the arm joints at the measured pose (grippers still
move); a command that reduces it is allowed, so an arm can always move back out.

- ee_bounds: {"left"|"right": [[xmin,xmax],[ymin,ymax],[zmin,zmax]]} for `tool0`, each in that arm's
  base frame (m). A violating arm holds on its own.
- min_arm_distance + right_base_in_left (4x4, right base frame -> left base frame): the closest
  pair among {wrist, flange, tool0} of the two arms must stay >= min_arm_distance (point-to-point,
  not surface). On a violation both arms hold.

Values are recorded on the robot with `python -m client.dk1_record_safety` and stored in
configs/task/dk1_hardware.json.
"""

from __future__ import annotations

import numpy as np

SIDES = ("left", "right")
ARM_JOINTS = {"left": slice(0, 6), "right": slice(7, 13)}
# Wrist (joint-5 axis), gripper flange, tool0 (between the fingertips).
SAFETY_BODIES = ("link5-6", "link6-7", "tool0")


class DK1Points:
    """FK of SAFETY_BODIES for one 6-DoF DK1 arm, in its base frame."""

    def __init__(self):
        import mujoco

        from client.teleop.mink_ik import load_dk1_arm_model

        self._mj = mujoco
        self.model = load_dk1_arm_model()
        self.data = mujoco.MjData(self.model)
        self.ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in SAFETY_BODIES]
        if min(self.ids) < 0:
            raise RuntimeError(f"DK1 model is missing one of {SAFETY_BODIES}")

    def __call__(self, q6) -> np.ndarray:
        """(len(SAFETY_BODIES), 3); the last row is tool0."""
        self.data.qpos[:] = np.asarray(q6, dtype=float).reshape(6)
        self._mj.mj_kinematics(self.model, self.data)
        return self.data.xpos[self.ids].copy()


def to_left(T_right_in_left: np.ndarray, pts_right: np.ndarray) -> np.ndarray:
    return pts_right @ T_right_in_left[:3, :3].T + T_right_in_left[:3, 3]


def fit_rigid(p_right: np.ndarray, p_left: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch: 4x4 T with p_left ~= R p_right + t. Returns (T, per-point residual in m)."""
    p_right, p_left = np.asarray(p_right, float), np.asarray(p_left, float)
    cr, cl = p_right.mean(0), p_left.mean(0)
    U, _, Vt = np.linalg.svd((p_right - cr).T @ (p_left - cl))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cl - R @ cr
    return T, np.linalg.norm(to_left(T, p_right) - p_left, axis=1)


class DK1Safety:
    def __init__(self, ee_bounds=None, right_base_in_left=None, min_arm_distance=None):
        self.ee_bounds = None if ee_bounds is None else {
            s: np.asarray(ee_bounds[s], float).reshape(3, 2) for s in SIDES}
        self.T = None if right_base_in_left is None else np.asarray(right_base_in_left, float).reshape(4, 4)
        self.min_dist = None if (min_arm_distance is None or self.T is None) else float(min_arm_distance)
        self._fk = None
        self.counts = {"box_left": 0, "box_right": 0, "arm_distance": 0}

    @property
    def active(self) -> bool:
        return self.ee_bounds is not None or self.min_dist is not None

    def fk(self) -> DK1Points:
        if self._fk is None:
            self._fk = DK1Points()
        return self._fk

    def points(self, q14) -> dict:
        q14 = np.asarray(q14, float)
        return {s: self.fk()(q14[ARM_JOINTS[s]]) for s in SIDES}

    def box_excess(self, side: str, ee) -> float:
        box = self.ee_bounds[side]
        return float((np.maximum(box[:, 0] - ee, 0) + np.maximum(ee - box[:, 1], 0)).sum())

    def arm_distance(self, pts: dict) -> float:
        right = to_left(self.T, pts["right"])
        return float(np.min(np.linalg.norm(pts["left"][:, None, :] - right[None, :, :], axis=-1)))

    def apply(self, action14, current14) -> np.ndarray:
        """Hold the arm joints that a command would push further into a violation."""
        if not self.active:
            return action14
        a = np.asarray(action14, float).copy()
        cmd, cur = self.points(a), self.points(current14)
        if self.ee_bounds is not None:
            for s in SIDES:
                new = self.box_excess(s, cmd[s][-1])
                if new > 0 and new >= self.box_excess(s, cur[s][-1]):
                    a[ARM_JOINTS[s]] = np.asarray(current14, float)[ARM_JOINTS[s]]
                    cmd[s] = cur[s]
                    self.counts[f"box_{s}"] += 1
        if self.min_dist is not None:
            new = self.arm_distance(cmd)
            if new < self.min_dist and new < self.arm_distance(cur):
                for s in SIDES:
                    a[ARM_JOINTS[s]] = np.asarray(current14, float)[ARM_JOINTS[s]]
                self.counts["arm_distance"] += 1
        return a

    def summary_and_reset(self) -> str:
        s = ", ".join(f"{k} {v}" for k, v in self.counts.items())
        self.counts = {k: 0 for k in self.counts}
        return s


def check_demos(safety: DK1Safety, demo_glob: str) -> dict:
    """Run recorded demos (state = current, action14 = command) through `safety`.

    Returns held-step counts and the demos' extremes (tool0 ranges per arm, min arm distance).
    """
    import glob

    import h5py

    paths = sorted(glob.glob(demo_glob))
    if not paths:
        raise FileNotFoundError(f"no demos match {demo_glob}")
    held = steps = 0
    lo = {s: np.full(3, np.inf) for s in SIDES}
    hi = {s: np.full(3, -np.inf) for s in SIDES}
    min_d = np.inf
    for p in paths:
        with h5py.File(p, "r") as f:
            st = f["saved_observation"]["state"][:].astype(float)
            act = f["action"]["action14"][:].astype(float)
        for t in range(len(act)):
            out = safety.apply(act[t], st[t])
            held += int(not np.allclose(out, act[t]))
            pts = safety.points(act[t])
            for s in SIDES:
                lo[s], hi[s] = np.minimum(lo[s], pts[s][-1]), np.maximum(hi[s], pts[s][-1])
            if safety.T is not None:
                min_d = min(min_d, safety.arm_distance(pts))
        steps += len(act)
    return {"episodes": len(paths), "steps": steps, "held": held, "counts": dict(safety.counts),
            "tool0_range": {s: np.stack([lo[s], hi[s]], 1) for s in SIDES},
            "min_arm_distance": None if safety.T is None else min_d}
