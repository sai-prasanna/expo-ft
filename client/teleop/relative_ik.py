"""Anchored end-effector-relative interventions from the DK1 leader arms.

Port of amazon-far/abc (d0832d1) `RelativeGelloController` + `relative_pose.py`:

  on clutch-in (per arm):  anchor_leader   = FK(leader joints)     (follower model)
                           anchor_follower = FK(follower joints)
  every tick:              target_pos = anchor_follower_pos + (leader_pos - anchor_leader_pos) * pos_scale
                           target_rot = scale(q_leader * q_anchor_leader^-1, rot_scale) * q_anchor_follower
                           joints     = IK(target, seed=previous command)

The rotation delta is applied in the world (arm base) frame. Grippers: on engage the follower
keeps its own grip until the leader trigger reaches it (no jump that would drop a held object),
then follows the trigger (0 open .. 1 closed) directly. Switching policy <-> human is a hard
switch, no blending.

`InterventionThread` runs leader reads + IK at ~60 Hz in the background and keeps the latest
14-D target, so the IK stays smooth and its seed never goes stale; `run_client` only sees
`step` at `control_hz` and just picks up the latest target.

14-D layout everywhere: [L j1..j6, L grip, R j1..j6, R grip].
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

from client.teleop.mink_ik import MinkArmIK

SIDES = ("left", "right")


def _slice(side: str) -> slice:
    return slice(0, 7) if side == "left" else slice(7, 14)


# ---------------------------------------------------------------- quaternion helpers (wxyz)
def _normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    return np.array([1.0, 0.0, 0.0, 0.0]) if n <= 1e-12 else q / n


def _multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _inverse(q: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize(q)
    return np.array([w, -x, -y, -z])


def _scaled_rotation(q: np.ndarray, scale: float) -> np.ndarray:
    q = _normalize(q)
    if q[0] < 0:
        q = -q
    angle = 2.0 * np.arccos(np.clip(q[0], -1.0, 1.0))
    s = np.sin(angle / 2.0)
    if abs(s) < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = q[1:] / s
    scaled = angle * float(scale)
    return _normalize(np.concatenate(([np.cos(scaled / 2.0)], axis * np.sin(scaled / 2.0))))


def quat_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic angle (rad) between two wxyz quaternions."""
    d = abs(float(np.dot(_normalize(a), _normalize(b))))
    return 2.0 * np.arccos(np.clip(d, -1.0, 1.0))


def apply_relative_pose_delta(
    *, tracked_pos, tracked_wxyz, anchor_tracked_pos, anchor_tracked_wxyz,
    anchor_output_pos, anchor_output_wxyz, position_scale=1.0, rotation_scale=1.0,
) -> tuple[np.ndarray, np.ndarray]:
    delta_pos = (np.asarray(tracked_pos, float) - np.asarray(anchor_tracked_pos, float)) * float(position_scale)
    delta_rot = _scaled_rotation(_multiply(_normalize(tracked_wxyz), _inverse(anchor_tracked_wxyz)), rotation_scale)
    out_rot = _multiply(delta_rot, _normalize(anchor_output_wxyz))
    return np.asarray(anchor_output_pos, float) + delta_pos, _normalize(out_rot)


# ---------------------------------------------------------------- controller
class RelativeIKController:
    """Maps anchored leader motion to absolute 14-D follower joint targets."""

    def __init__(
        self,
        *,
        home14: np.ndarray | None = None,
        position_scale: float = 1.0,
        rotation_scale: float = 1.0,
        leader_joint_offsets: np.ndarray | None = None,
        leader_joint_signs: np.ndarray | None = None,
        **ik_kwargs,
    ):
        home14 = None if home14 is None else np.asarray(home14, float).reshape(14)
        self.solvers = {
            side: MinkArmIK(None if home14 is None else home14[_slice(side)][:6], **ik_kwargs) for side in SIDES
        }
        self.position_scale = float(position_scale)
        self.rotation_scale = float(rotation_scale)
        # leader joint -> follower joint convention: q_f = sign * q_l + offset (per arm, 6 joints, 12 total)
        self.offsets = np.zeros(12) if leader_joint_offsets is None else np.asarray(leader_joint_offsets, float).reshape(12)
        self.signs = np.ones(12) if leader_joint_signs is None else np.asarray(leader_joint_signs, float).reshape(12)
        self.anchors: dict[str, dict[str, np.ndarray]] = {}
        self.last = np.zeros(14)
        self.grip_latch_tol = 0.05
        self._grip = {}  # side -> dict(hold, start, latched)

    def leader_to_follower_joints(self, leader14: np.ndarray) -> np.ndarray:
        """Apply the leader->follower joint map to the 6 arm joints of each side (grippers untouched)."""
        out = np.asarray(leader14, float).reshape(14).copy()
        out[0:6] = self.signs[0:6] * out[0:6] + self.offsets[0:6]
        out[7:13] = self.signs[6:12] * out[7:13] + self.offsets[6:12]
        return out

    def anchor(self, leader14: np.ndarray, follower14: np.ndarray) -> None:
        leader14 = self.leader_to_follower_joints(leader14)
        follower14 = np.asarray(follower14, float).reshape(14)
        self.last = follower14.copy()
        self.anchors.clear()
        for side in SIDES:
            sl = _slice(side)
            t_pos, t_wxyz = self.solvers[side].fk(leader14[sl][:6])
            o_pos, o_wxyz = self.solvers[side].fk(follower14[sl][:6])
            self.anchors[side] = dict(tracked_pos=t_pos, tracked_wxyz=t_wxyz, output_pos=o_pos, output_wxyz=o_wxyz)
            hold = float(np.clip(follower14[sl][6], 0.0, 1.0))
            start = float(np.clip(leader14[sl][6], 0.0, 1.0))
            self._grip[side] = dict(hold=hold, start=start, latched=abs(start - hold) <= self.grip_latch_tol)

    def _gripper(self, side: str, trigger: float) -> float:
        """Keep the follower's grip until the trigger crosses it, then track the trigger."""
        trigger = float(np.clip(trigger, 0.0, 1.0))
        g = self._grip[side]
        if not g["latched"]:
            crossed = (trigger - g["hold"]) * (g["start"] - g["hold"]) <= 0
            if crossed or abs(trigger - g["hold"]) <= self.grip_latch_tol:
                g["latched"] = True
            else:
                return g["hold"]
        return trigger

    def targets(self, leader14: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        leader14 = self.leader_to_follower_joints(leader14)
        out = {}
        for side in SIDES:
            t_pos, t_wxyz = self.solvers[side].fk(leader14[_slice(side)][:6])
            a = self.anchors[side]
            out[side] = apply_relative_pose_delta(
                tracked_pos=t_pos, tracked_wxyz=t_wxyz,
                anchor_tracked_pos=a["tracked_pos"], anchor_tracked_wxyz=a["tracked_wxyz"],
                anchor_output_pos=a["output_pos"], anchor_output_wxyz=a["output_wxyz"],
                position_scale=self.position_scale, rotation_scale=self.rotation_scale,
            )
        return out

    def step(self, leader14: np.ndarray) -> np.ndarray:
        if not self.anchors:
            raise RuntimeError("RelativeIKController.step() before anchor()")
        result = self.last.copy()
        targets = self.targets(leader14)
        leader14 = np.asarray(leader14, float).reshape(14)
        for side in SIDES:
            sl = _slice(side)
            pos, wxyz = targets[side]
            arm = result[sl]  # view
            arm[:6] = self.solvers[side].solve(pos, wxyz, arm[:6])
            arm[6] = self._gripper(side, leader14[sl][6])
        self.last = result
        return result.copy()


# ---------------------------------------------------------------- background thread
class InterventionThread:
    """Reads the leaders + runs the IK at `rate_hz` while the clutch is engaged.

    read_leader():   -> 14-D leader joints (radians) + triggers (0..1)
    read_follower(): -> 14-D measured follower joints + grippers
    """

    def __init__(
        self,
        controller: RelativeIKController,
        clutch,
        read_leader: Callable[[], np.ndarray],
        read_follower: Callable[[], np.ndarray],
        rate_hz: float = 60.0,
        max_consecutive_errors: int = 6,
    ):
        self.controller = controller
        self.clutch = clutch
        self.read_leader = read_leader
        self.read_follower = read_follower
        self.period = 1.0 / float(rate_hz)
        self._lock = threading.Lock()
        self._target: np.ndarray | None = None
        self._active = False
        self._anchored_epoch = -1
        self._stop = threading.Event()
        self.last_error: Exception | None = None
        self.max_consecutive_errors = int(max_consecutive_errors)
        self._errors = 0
        self.loop_ms = 0.0
        self._thread = threading.Thread(target=self._loop, daemon=True, name="dk1-intervention")
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self._tick()
                self._errors = 0
            except Exception as e:  # never kill the thread; tolerate transient read errors
                self.last_error = e
                self._errors += 1
                if self._errors < self.max_consecutive_errors:
                    print(f"[intervention] transient error {self._errors}: {e!r} (keeping last target)", flush=True)
                else:
                    print(f"\n[intervention] !!! {self._errors} consecutive errors ({e!r}); "
                          f"RELEASING TO POLICY CONTROL !!!\n", flush=True)
                    with self._lock:
                        self._active = False
                        self._target = None
                    self._anchored_epoch = -1
                    self._errors = 0
                    self.clutch.set_engaged(False)
            self.loop_ms = (time.perf_counter() - t0) * 1000.0
            time.sleep(max(0.0, self.period - (time.perf_counter() - t0)))

    def _tick(self):
        engaged, epoch = self.clutch.state()
        if not engaged:
            with self._lock:
                self._active = False
                self._target = None
            return
        leader = self.read_leader()
        if epoch != self._anchored_epoch:
            self.controller.anchor(leader, self.read_follower())
            self._anchored_epoch = epoch
        target = self.controller.step(leader)
        with self._lock:
            # The leader read + IK take a while: drop the result if the clutch was released or
            # re-engaged (new anchor needed) meanwhile, so get() never reports a stale human action.
            if self.clutch.state() != (True, epoch):
                self._active = False
                self._target = None
                return
            self._target = target
            self._active = True

    def get(self) -> tuple[np.ndarray | None, bool]:
        if not self.clutch.engaged:
            return None, False
        with self._lock:
            if not self._active or self._target is None:
                return None, False
            return self._target.copy(), True

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
