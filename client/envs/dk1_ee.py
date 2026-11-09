"""DROID-style end-effector action space for the DK1 (EXPO-FT "cartesian_velocity", per arm).

Per arm, in that arm's base frame (FK of the DK1 follower URDF, `tool0` frame):

    action (7) = [dpos / MAX_LIN_DELTA, deuler_xyz / MAX_ROT_DELTA, gripper]
        dpos   = p_target - p_current
        deuler = (R_target * R_current^-1).as_euler("xyz")   (rotation delta in the base frame)
        gripper: absolute 0 (open) .. 1 (closed), as in the joint action space
    state  (10) = [pos(3), rot6d(6), gripper]
        rot6d = first two columns of R (continuous; Euler wraps: the DK1 tool sits at roll ~ -pi)

This is DROID's `cartesian_velocity` convention (droid/robot_ik/robot_ik_solver.py: velocity =
pose delta / max delta per 15 Hz step, applied with `add_poses`). Unlike DROID, the velocity is
NOT norm-clipped to 1: the per-step joint cap downstream is the safety limit (same as for the
joint policy), so training data stays lossless.

    14-D action: [L(7), R(7)]       20-D state: [L(10), R(10)]

Training data: target = FK(action14 joints) (the commanded target), current = FK(state joints)
(measured). Inference: target pose = current pose (+) action, joints = IK(target, seed = measured
joints), then the env's usual joint safety stack (joint limits, per-step cap, command rate limit,
EE box, arm distance).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

MAX_LIN_DELTA = 0.075  # m per step at 15 Hz (DROID RobotIKSolver.max_lin_delta)
MAX_ROT_DELTA = 0.15  # rad per step (DROID RobotIKSolver.max_rot_delta)
EE_ACTION_DIM = 14
EE_STATE_DIM = 20
SIDES = ("left", "right")


def _jslice(side):  # 14-D joint layout [L j1..j6, L grip, R j1..j6, R grip]
    return slice(0, 7) if side == "left" else slice(7, 14)


def _rot(wxyz) -> Rotation:
    w, x, y, z = np.asarray(wxyz, dtype=float)
    return Rotation.from_quat([x, y, z, w])


def _wxyz(r: Rotation) -> np.ndarray:
    x, y, z, w = r.as_quat()
    return np.array([w, x, y, z])


class DK1EEConverter:
    """Joint <-> EE conversions for both arms (both arms share the follower model)."""

    def __init__(self):
        from client.teleop.mink_ik import DK1Kinematics

        self.kin = DK1Kinematics()

    def pose(self, q6) -> tuple[np.ndarray, Rotation]:
        p, q = self.kin.fk(q6)
        return p, _rot(q)

    def state(self, joints14) -> np.ndarray:
        joints14 = np.asarray(joints14, dtype=float).reshape(14)
        out = []
        for side in SIDES:
            q = joints14[_jslice(side)]
            p, r = self.pose(q[:6])
            m = r.as_matrix()
            out.append(np.concatenate([p, m[:, 0], m[:, 1], [q[6]]]))
        return np.concatenate(out).astype(np.float32)

    def action(self, target14, current14) -> np.ndarray:
        """EE action that moves the measured `current14` pose to the commanded `target14` pose."""
        target14 = np.asarray(target14, dtype=float).reshape(14)
        current14 = np.asarray(current14, dtype=float).reshape(14)
        out = []
        for side in SIDES:
            t, c = target14[_jslice(side)], current14[_jslice(side)]
            pt, rt = self.pose(t[:6])
            pc, rc = self.pose(c[:6])
            dpos = (pt - pc) / MAX_LIN_DELTA
            drot = (rt * rc.inv()).as_euler("xyz") / MAX_ROT_DELTA
            out.append(np.concatenate([dpos, drot, [t[6]]]))
        return np.concatenate(out).astype(np.float32)

    def target_poses(self, action14, current14):
        """{side: (target_pos, target_wxyz, gripper)} for an EE action applied at `current14`."""
        action14 = np.asarray(action14, dtype=float).reshape(14)
        current14 = np.asarray(current14, dtype=float).reshape(14)
        out = {}
        for side in SIDES:
            a, c = action14[_jslice(side)], current14[_jslice(side)]
            pc, rc = self.pose(c[:6])
            pt = pc + a[:3] * MAX_LIN_DELTA
            rt = Rotation.from_euler("xyz", a[3:6] * MAX_ROT_DELTA) * rc
            out[side] = (pt, _wxyz(rt), float(a[6]))
        return out


class DK1EEIK:
    """EE action -> 14-D joint target. Solves to convergence (not the 60 Hz teleop tracker):
    seed = measured joints, Gauss-Newton-like steps (gain 1), no joint velocity limit, no posture
    pull worth mentioning. The joint cap downstream bounds the motion."""

    def __init__(self, iterations: int = 15, max_joint_delta: float = 0.6):
        from client.teleop.mink_ik import MinkArmIK

        self.conv = DK1EEConverter()
        self.ik = {
            side: MinkArmIK(dt=1.0, iterations=iterations, gain=1.0, max_velocity=0.0, limit_margin=0.0,
                            position_cost=50.0, orientation_cost=10.0, posture_cost=1e-4,
                            max_joint_delta=max_joint_delta)
            for side in SIDES
        }

    def __call__(self, action14, current14) -> np.ndarray:
        current14 = np.asarray(current14, dtype=float).reshape(14)
        targets = self.conv.target_poses(action14, current14)
        joints = current14.copy()
        for side in SIDES:
            pos, wxyz, grip = targets[side]
            sl = _jslice(side)
            joints[sl][:6] = self.ik[side].solve(pos, wxyz, seed=current14[sl][:6])
            joints[sl][6] = grip
        return joints


def make_ee_env(base_cls):
    """Subclass of a DK1 joint-space env (DK1Env, DK1ReplayEnv) that talks EE actions/state.

    obs["state"] is the 20-D EE state (joints stay in obs["joint_state"]). step() takes a 14-D EE
    action, IKs it to joints and runs the base env's joint step; executed_action is converted back
    to the EE action of the joint command actually sent (after clipping), so RL buffers store
    what was executed. get_human_action() (leader interventions) is converted the same way.
    An all -1 action stays the "hold" sentinel.
    """

    class DK1EEEnv(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._ee_ik = DK1EEIK()
            self._ee = self._ee_ik.conv
            self._ee_joints = None  # measured joints of the latest observation

        def _measured_joints(self):
            if hasattr(self, "get_state"):
                return np.asarray(self.get_state(), dtype=float)
            return self._ee_joints

        def transform_observation(self, raw):
            obs = super().transform_observation(raw)
            return self._to_ee_obs(obs)

        def get_observation(self):
            obs = super().get_observation()
            if "joint_state" not in obs:  # base classes without transform_observation (replay env)
                obs = self._to_ee_obs(obs)
            return obs

        def _to_ee_obs(self, obs):
            joints = np.asarray(obs["state"], dtype=float)
            self._ee_joints = joints
            obs = dict(obs)
            obs["joint_state"] = joints.astype(np.float32)
            obs["state"] = self._ee.state(joints)
            return obs

        def step(self, action, max_delta=None):
            action = np.asarray(action, dtype=np.float64).reshape(-1)[:14]
            current = self._measured_joints()
            if current is None:
                raise RuntimeError("DK1EEEnv.step before any observation")
            hold = np.allclose(action, -1.0)
            joints = action if hold else self._ee_ik(action, current)
            result = dict(super().step(joints, max_delta=max_delta))
            executed = np.asarray(result["executed_action"], dtype=float)
            result["executed_joints"] = executed
            result["executed_action"] = self._ee.action(executed, current)
            return result

        def get_human_action(self):
            joints, is_human = super().get_human_action()
            if not is_human or joints is None:
                return joints, is_human
            return self._ee.action(joints, self._measured_joints()), is_human

    DK1EEEnv.__name__ = DK1EEEnv.__qualname__ = f"{base_cls.__name__}EE"
    return DK1EEEnv
