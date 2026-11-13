"""End-effector action space for the DK1 ("ee_cmd"), per arm in that arm's base frame (FK of the
DK1 follower URDF, `tool0` frame). Same layout as EXPO-FT's DROID "cartesian_velocity":

    action (7) = [dpos / MAX_LIN_DELTA, deuler_xyz / MAX_ROT_DELTA, gripper]
    state  (10) = [pos(3), rot6d(6), gripper]      rot6d = first two columns of R
    14-D action: [L(7), R(7)]       20-D state: [L(10), R(10)]

Every action is relative to the ANCHOR: the last joint command sent (at the first step of an
episode: the measured joints). So an action is how far the commanded hand moves this step, and
the arms' tracking lag never enters the labels.

    training  (demos: measured q_t, executed command c_t):
        anchor_t = c_{t-1}  (anchor_0 = q_0)
        action_t = EE(c_t) (-) EE(anchor_t)      dpos = p_c - p_a, deuler = (R_c R_a^-1).as_euler("xyz")
        state_t  = EE(anchor_t), gripper = measured
    rollout  (EnvClientWrapper, server side):
        joints = IK(EE(anchor) (+) action, seed = anchor);  the client applies its joint safety;
        anchor <- the joint command the client executed (covers clipping, holds, interventions)

gripper: absolute 0 (open) .. 1 (closed), as in the joint action space. Actions are not clipped
(rotation labels exceed 1 on fast wrist turns); the client's per-step joint cap is the limit.
Robot replay checks: scripts/dk1_task/ee_label_check/robot_replay.py (2026-11-13: matches joint
replay; the old measured-anchored labels drifted 4 cm in 4 s on the same demo).
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

    def state(self, anchor14, measured14) -> np.ndarray:
        """20-D EE state: pose of the anchor joints, measured grippers."""
        anchor14 = np.asarray(anchor14, dtype=float).reshape(14)
        measured14 = np.asarray(measured14, dtype=float).reshape(14)
        out = []
        for side in SIDES:
            sl = _jslice(side)
            p, r = self.pose(anchor14[sl][:6])
            m = r.as_matrix()
            out.append(np.concatenate([p, m[:, 0], m[:, 1], [measured14[sl][6]]]))
        return np.concatenate(out).astype(np.float32)

    def action(self, target14, anchor14) -> np.ndarray:
        """EE action that moves the anchor pose to the pose of the joint command `target14`."""
        target14 = np.asarray(target14, dtype=float).reshape(14)
        anchor14 = np.asarray(anchor14, dtype=float).reshape(14)
        out = []
        for side in SIDES:
            t, a = target14[_jslice(side)], anchor14[_jslice(side)]
            pt, rt = self.pose(t[:6])
            pa, ra = self.pose(a[:6])
            dpos = (pt - pa) / MAX_LIN_DELTA
            drot = (rt * ra.inv()).as_euler("xyz") / MAX_ROT_DELTA
            out.append(np.concatenate([dpos, drot, [t[6]]]))
        return np.concatenate(out).astype(np.float32)

    def joints(self, action14, anchor14) -> np.ndarray:
        """14-D joint command of an EE action applied to the anchor (IK seeded with the anchor)."""
        if self._ik is None:
            self._ik = self._make_ik()
        action14 = np.asarray(action14, dtype=float).reshape(14)
        anchor14 = np.asarray(anchor14, dtype=float).reshape(14)
        joints = anchor14.copy()
        for side in SIDES:
            a, c = action14[_jslice(side)], anchor14[_jslice(side)]
            pa, ra = self.pose(c[:6])
            pt = pa + a[:3] * MAX_LIN_DELTA
            rt = Rotation.from_euler("xyz", a[3:6] * MAX_ROT_DELTA) * ra
            sl = _jslice(side)
            joints[sl][:6] = self._ik[side].solve(pt, _wxyz(rt), seed=c[:6])
            joints[sl][6] = a[6]
        return joints

    def episode(self, measured, commands) -> tuple[np.ndarray, np.ndarray]:
        """Demo (measured q (T,14), executed commands c (T,14)) -> (states (T,20), actions (T,14))."""
        measured = np.asarray(measured, dtype=float)
        commands = np.asarray(commands, dtype=float)
        anchors = np.concatenate([measured[:1], commands[:-1]])
        states = np.stack([self.state(a, q) for a, q in zip(anchors, measured)])
        actions = np.stack([self.action(c, a) for c, a in zip(commands, anchors)])
        return states, actions
