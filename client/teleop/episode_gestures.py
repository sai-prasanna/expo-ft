"""Hands-free episode control for leader-arm teleop: auto start/stop + trigger keep/discard gesture.

Episodes go home -> task -> home, and the end of one episode is the start of the next (no
homing in between). ABC-130k operators do the same: they start from a rest pose and bring the
arms back to it before saving.

"Home" is loose on purpose: only the shoulder/elbow joints (j1-j3 of each arm) are compared,
within stop_radius. Wrist angles don't matter.

    IDLE    Nothing is recorded; take as long as you like to reset the scene. Once the arms are
            home, still, and the triggers open, the resting pose is tracked (it follows slow
            drift while the arms are still). Clearly moving an arm, i.e. any joint more than
            start_radius from that resting pose -> "start". Triggers never start an episode.
    RECORD  Once the arms went > arm_radius from home, coming back home and holding still for
            hold_s -> "stop" (the still frames are part of the episode). Before that, being
            home does nothing: recording just continues.
    DECIDE  Recording is over; waits for the operator, ignoring arm motion. Squeeze (> grip_closed)
            and hold for gesture_s: both triggers -> "keep", LEFT trigger alone -> "discard",
            right alone -> nothing. Then back to IDLE.

An episode is never dropped automatically; only the gesture (or 1/2 + Enter) ends it.
All poses are 12 arm joints in follower space [L j1-6, R j1-6]; grips are leader triggers in
[0, 1], 1 = closed. home=None uses the IDLE resting pose as home.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Joints compared against home: shoulder/elbow j1-j3 of each arm (indices into the 12 arm joints).
HOME_JOINTS = (0, 1, 2, 6, 7, 8)


@dataclass
class GestureParams:
    start_radius: float = 0.1  # rad, moving any joint this far from the resting pose starts recording
    arm_radius: float = 0.6  # rad (j1-j3), the episode can only auto-stop after going this far from home
    stop_radius: float = 0.4  # rad (j1-j3), within this of home counts as "home"
    still_speed: float = 0.2  # rad/s, max joint speed (all joints) that counts as "still"
    hold_s: float = 0.5  # stillness at home that ends an episode
    ready_s: float = 0.5  # stillness at home (triggers open) before a start can be detected
    grip_open: float = 0.3  # below = open
    grip_closed: float = 0.5  # above = squeezed
    gesture_s: float = 0.6  # how long a squeeze must be held


class EpisodeGestureFSM:
    def __init__(self, hz: float, params: GestureParams | None = None, home=None):
        self.p = params or GestureParams()
        self.hz = float(hz)
        self.home = None if home is None else np.asarray(home, dtype=np.float64).reshape(12)
        self._home_idx = np.asarray(HOME_JOINTS)
        self.home_fallback = None  # home=None: the first resting pose
        self.reset()

    def _steps(self, seconds: float) -> int:
        return max(1, int(round(seconds * self.hz)))

    def reset(self):
        """Back to IDLE (not ready yet)."""
        self.state = "idle"
        self.ready = False
        self.ref = None  # resting pose in IDLE
        self._prev = None
        self._count = 0
        self._armed = False
        self._pattern = None
        self._pattern_count = 0
        self.last = {}

    @property
    def armed(self) -> bool:
        return self._armed

    def update(self, q12, grips) -> str | None:
        """Feed one control step. Returns "ready", "start", "stop", "keep", "discard" or None."""
        q = np.asarray(q12, dtype=np.float64).reshape(12)
        g = np.asarray(grips, dtype=np.float64).reshape(2)
        vel = np.zeros(12) if self._prev is None else np.abs(q - self._prev) * self.hz
        self._prev = q
        speed = float(vel.max())
        still = speed < self.p.still_speed
        p = self.p
        if self.home is not None:
            home = self.home
        else:
            home = self.home_fallback if self.home_fallback is not None else q
        dev_home = np.zeros(12)
        dev_home[self._home_idx] = np.abs(q - home)[self._home_idx]
        dist_home = float(dev_home.max())
        # For the collector's status line.
        self.last = {"speed": speed, "fast_joint": int(vel.argmax()), "grips": g, "q": q, "home": home,
                     "dist": dist_home, "far_joint": int(dev_home.argmax()), "moved": None}

        if self.state == "idle":
            if not self.ready:
                ok = still and dist_home < p.stop_radius and bool(np.all(g < p.grip_open))
                self._count = self._count + 1 if ok else 0
                if self._count >= self._steps(p.ready_s):
                    self.ready, self.ref, self._count = True, q.copy(), 0
                    if self.home is None:
                        self.home_fallback = q.copy()
                    return "ready"
                return None
            moved = float(np.max(np.abs(q - self.ref)))
            self.last["moved"] = moved
            if moved > p.start_radius:
                self.state, self._count, self._armed = "record", 0, False
                return "start"
            if still:
                self.ref = q.copy()  # follow slow drift / sag while nobody is moving the arms
            return None

        if self.state == "record":
            if dist_home > p.arm_radius:
                self._armed = True
            back = self._armed and dist_home < p.stop_radius and still
            self._count = self._count + 1 if back else 0
            if self._count >= self._steps(p.hold_s):
                self.state, self._count = "decide", 0
                self._pattern, self._pattern_count = None, 0
                return "stop"
            return None

        if self.state == "decide":
            closed = g > p.grip_closed
            pattern = "both" if closed.all() else ("left" if closed[0] else None)
            if pattern != self._pattern:
                self._pattern, self._pattern_count = pattern, 0
            if pattern is not None:
                self._pattern_count += 1
                if self._pattern_count >= self._steps(p.gesture_s):
                    self.reset()
                    return "keep" if pattern == "both" else "discard"
            return None

        raise RuntimeError(f"bad state {self.state}")
