"""DROID-style end-effector action space for the DK1 (EXPO-FT "cartesian_velocity", per arm).

Per arm, in that arm's base frame (FK of the DK1 follower URDF, `tool0` frame):

    action (7) = [dpos / MAX_LIN_DELTA, deuler_xyz / MAX_ROT_DELTA, gripper]
        dpos   = p_target - p_current
        deuler = (R_target * R_current^-1).as_euler("xyz")   (rotation delta in the base frame)
        gripper: absolute 0 (open) .. 1 (closed), as in the joint action space
    state  (10) = [pos(3), rot6d(6), gripper]
        rot6d = first two columns of R (continuous; Euler wraps: the DK1 tool sits at roll ~ -pi)

    14-D action: [L(7), R(7)]       20-D state: [L(10), R(10)]

Training data (converter, demo loader): target = FK(commanded joints c_t), current = FK(measured
joints q_t) of the same step. Rollouts (EnvClientWrapper, server side): target = FK(q_t of the
latest observation) (+) action, joints = IK(target, seed = q_t). The robot client only ever sees
joint targets and applies its usual joint safety. Velocities are NOT norm-clipped as in DROID: the
client's per-step joint cap is the safety limit.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

MAX_LIN_DELTA = 0.075  # m per step at 15 Hz (DROID RobotIKSolver.max_lin_delta)
MAX_ROT_DELTA = 0.15  # rad per step (DROID RobotIKSolver.max_rot_delta)
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


class DK1EE:
    """Joint <-> EE conversions for both arms (both arms share the follower model)."""

    def __init__(self):
        from client.teleop.mink_ik import DK1Kinematics, MinkArmIK

        self.kin = DK1Kinematics()
        # IK solved to convergence (not the 60 Hz teleop tracker): seed = measured joints,
        # Gauss-Newton-like steps (gain 1), no velocity limit. The client's joint cap bounds motion.
        self._ik = None
        self._make_ik = lambda: {
            side: MinkArmIK(dt=1.0, iterations=15, gain=1.0, max_velocity=0.0, limit_margin=0.0,
                            position_cost=50.0, orientation_cost=10.0, posture_cost=1e-4,
                            max_joint_delta=0.6)
            for side in SIDES
        }

    def pose(self, q6) -> tuple[np.ndarray, Rotation]:
        p, q = self.kin.fk(q6)
        return p, _rot(q)

    def state(self, joints14) -> np.ndarray:
        """20-D EE state of the measured joints."""
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

    def joints(self, action14, current14) -> np.ndarray:
        """14-D joint target of an EE action applied at the measured `current14` (IK)."""
        if self._ik is None:
            self._ik = self._make_ik()
        action14 = np.asarray(action14, dtype=float).reshape(14)
        current14 = np.asarray(current14, dtype=float).reshape(14)
        joints = current14.copy()
        for side in SIDES:
            a, c = action14[_jslice(side)], current14[_jslice(side)]
            pc, rc = self.pose(c[:6])
            pt = pc + a[:3] * MAX_LIN_DELTA
            rt = Rotation.from_euler("xyz", a[3:6] * MAX_ROT_DELTA) * rc
            sl = _jslice(side)
            joints[sl][:6] = self._ik[side].solve(pt, _wxyz(rt), seed=c[:6])
            joints[sl][6] = a[6]
        return joints
