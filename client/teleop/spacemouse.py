"""Two SpaceMice (one per arm) as the DK1 intervention device, an alternative to pedal + GELLO.

Selected with task config `intervention.device = "spacemouse"` (settings in `intervention.spacemouse`).

- Engage: AUTO. Deflecting either puck past `engage_threshold`, or pressing a button, hands control
  to the human; after `idle_release_s` with every puck centered and no button held, control returns
  to the policy. Every engage re-anchors (target = current follower EE pose).
- Motion: the puck deflection is an EE velocity (linear `max_lin_vel` m/s, angular `max_rot_vel`
  rad/s at full deflection) in the operator frame [forward, left, up], rotated into the arm base
  frame by `operator_yaw_deg` (0 = operator faces the same way as the arms). It is integrated into an
  absolute EE target and solved with the same mink IK as the GELLO path. The target never leads the
  IK solution by more than `max_lead_m` / `max_lead_rad` (no wind-up at joint limits / the cap).
- Gripper (as EXPO-FT): button 0 HELD = close, released = open. Until button 0 is first pressed
  after an engage, the follower keeps its own grip (a nudge never drops a held object).
- Rotation (`rotation_mode`): "button1" (default) = translation only; HOLD button 1 = rotation only.
  Pushing a puck also tilts it a lot (push forward reads more pitch than forward), so mixing both
  would tip the gripper on every push. "always" = 6-DoF at once.

Pucks are matched by USB port (evdev `phys` prefix, `left_phys` / `right_phys`); list them with
`python -m client.teleop.spacemouse --list`. Needs read access: scripts/dk1_task/71-dk1-spacemouse.rules.

Dry check, no robot (prints what each arm would be told to do):
    python -m client.teleop.spacemouse
"""

from __future__ import annotations

import select
import threading
import time

import numpy as np

from client.teleop.mink_ik import MinkArmIK
from client.teleop.relative_ik import SIDES, _multiply, _normalize, _slice, quat_angle

DEFAULTS = dict(
    left_phys="usb-0000:00:14.0-3",
    right_phys="usb-0000:00:14.0-9.2",
    full_scale=350.0,  # raw |value| at full deflection (SpaceMouse Compact)
    # operator-frame [fwd, left, up, roll(about fwd), pitch(about left), yaw(about up)] from raw
    # evdev [REL_X, REL_Y, REL_Z, REL_RX, REL_RY, REL_RZ]: (index, sign) each.
    # Checked 2026-11-11 (SpaceMouse Compact, operator behind the arms): push away = +x, left = +y,
    # up = +z; tilts follow the same flips. Yaw (twist) sign not checked yet.
    axes=((1, 1), (0, 1), (2, -1), (4, 1), (3, 1), (5, -1)),
    operator_yaw_deg=0.0,
    deadzone=0.08,  # normalized; below this an axis reads 0
    engage_threshold=0.15,  # normalized deflection that takes over
    idle_release_s=0.75,
    stale_s=0.3,  # no event from a puck for this long -> treat it as centered
    max_lin_vel=0.08,  # m/s at full deflection
    max_rot_vel=0.5,  # rad/s at full deflection
    max_lead_m=0.02,
    max_lead_rad=0.15,
    grip_closed=1.0,
    grip_rate=3.0,  # gripper command slew, 1/s
    rotation_mode="button1",  # "button1": hold button 1 = rotate only, else translate only | "always"
)


def _cfg(cfg) -> dict:
    out = dict(DEFAULTS)
    if cfg:
        out.update({k: v for k, v in dict(cfg).items() if v is not None})
    return out


def list_spacemice() -> list:
    import evdev

    found = []
    for path in evdev.list_devices():
        try:
            d = evdev.InputDevice(path)
        except OSError:
            continue
        if evdev.ecodes.EV_REL in d.capabilities() and ("3dconnexion" in d.name.lower() or "spacemouse" in d.name.lower()):
            found.append(d)
        else:
            d.close()
    return found


# ---------------------------------------------------------------- device reader + engage gate
class SpaceMice:
    """Reads both pucks in a thread. Doubles as the intervention gate (same interface as Pedal:
    state() -> (engaged, epoch), engaged, set_engaged(), clicks(), close())."""

    def __init__(self, cfg=None, mock: bool = False):
        self.cfg = _cfg(cfg)
        c = self.cfg
        self._axes = [(int(i), float(s)) for i, s in c["axes"]]
        yaw = np.deg2rad(float(c["operator_yaw_deg"]))
        cy, sy = np.cos(yaw), np.sin(yaw)
        self._R = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])  # operator -> arm base
        self._lock = threading.Lock()
        self._raw = {s: np.zeros(6) for s in SIDES}
        self._t_event = {s: 0.0 for s in SIDES}
        self._buttons = {s: set() for s in SIDES}
        self._engaged = False
        self._epoch = 0
        self._clicks = 0
        self._last_active = 0.0
        self._stop = threading.Event()
        self._thread = None
        self._devs: dict = {}
        if mock:
            return
        self._devs = self._open()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="spacemice")
        self._thread.start()
        print(f"[spacemouse] L {self._devs['left'].path}, R {self._devs['right'].path}: move a puck to take over, "
              f"let go for {c['idle_release_s']} s to hand back; HOLD button 0 = close gripper.", flush=True)

    def _open(self) -> dict:
        devs = list_spacemice()
        out = {}
        for side in SIDES:
            want = str(self.cfg[f"{side}_phys"])
            match = [d for d in devs if (d.phys or "").split("/")[0] == want]
            if len(match) != 1:
                for d in devs:
                    d.close()
                have = ", ".join(f"{d.path} {d.phys}" for d in devs) or "none readable"
                raise FileNotFoundError(f"{side} SpaceMouse at {want!r} not found (have: {have}). Set "
                                        f"intervention.spacemouse.{side}_phys, or install scripts/dk1_task/71-dk1-spacemouse.rules")
            out[side] = match[0]
        for d in devs:
            if all(d is not o for o in out.values()):
                d.close()
        return out

    def _loop(self):
        import evdev

        E = evdev.ecodes
        code_idx = {E.REL_X: 0, E.REL_Y: 1, E.REL_Z: 2, E.REL_RX: 3, E.REL_RY: 4, E.REL_RZ: 5}
        while not self._stop.is_set():
            if not self._devs:
                try:
                    self._devs = self._open()
                    print("[spacemouse] reconnected", flush=True)
                except OSError:
                    self._stop.wait(1.0)
                    continue
            by_fd = {d.fd: (side, d) for side, d in self._devs.items()}
            try:
                for _, d in by_fd.values():
                    try:
                        d.grab()
                    except OSError:
                        pass
                while not self._stop.is_set():
                    for fd in select.select(list(by_fd), [], [], 0.02)[0]:
                        side, d = by_fd[fd]
                        for ev in d.read():
                            with self._lock:
                                if ev.type == E.EV_REL and ev.code in code_idx:
                                    self._raw[side][code_idx[ev.code]] = ev.value
                                    self._t_event[side] = time.monotonic()
                                elif ev.type == E.EV_KEY:
                                    if ev.value:
                                        self._buttons[side].add(ev.code)
                                    else:
                                        self._buttons[side].discard(ev.code)
                                        if not any(self._buttons.values()):
                                            self._clicks += 1
                    self._update_gate()
            except OSError as e:
                if not self._stop.is_set():
                    print(f"\n[spacemouse] !!! SpaceMouse lost ({e!r}); policy control !!!\n", flush=True)
                with self._lock:
                    self._raw = {s: np.zeros(6) for s in SIDES}
                    self._buttons = {s: set() for s in SIDES}
                self.set_engaged(False)
                for d in self._devs.values():
                    try:
                        d.close()
                    except OSError:
                        pass
                self._devs = {}
        for d in self._devs.values():
            try:
                d.ungrab()
                d.close()
            except OSError:
                pass

    # -------------------------------------------------------------- readings
    def _twist_locked(self, side: str) -> np.ndarray:
        """Normalized, deadzoned [v_fwd, v_left, v_up, w_x, w_y, w_z] in the arm base frame (-1..1)."""
        c = self.cfg
        if time.monotonic() - self._t_event[side] > c["stale_s"]:
            return np.zeros(6)
        raw = self._raw[side]
        op = np.array([s * raw[i] for i, s in self._axes]) / float(c["full_scale"])
        op = np.clip(op, -1.0, 1.0)
        dz = float(c["deadzone"])
        op = np.sign(op) * np.clip((np.abs(op) - dz) / (1.0 - dz), 0.0, 1.0)
        return np.concatenate([self._R @ op[:3], self._R @ op[3:]])

    def _update_gate(self):
        now = time.monotonic()
        with self._lock:
            peak = max(np.abs(self._twist_locked(s)).max() for s in SIDES)
            buttons = any(self._buttons.values())
            moving = peak > 0.0
            if buttons or peak >= self.cfg["engage_threshold"] or (self._engaged and moving):
                self._last_active = now
                if not self._engaged and (buttons or peak >= self.cfg["engage_threshold"]):
                    self._set(True)
            elif self._engaged and now - self._last_active > self.cfg["idle_release_s"]:
                self._set(False)

    def read(self) -> dict:
        """{side: (twist6 normalized in the arm base frame, button0 held)}; rotation_mode applied."""
        with self._lock:
            out = {}
            for s in SIDES:
                tw = self._twist_locked(s)
                if self.cfg["rotation_mode"] == "button1":
                    if 257 in self._buttons[s]:
                        tw[:3] = 0.0
                    else:
                        tw[3:] = 0.0
                out[s] = (tw, 256 in self._buttons[s])
            return out

    # -------------------------------------------------------------- gate interface
    def set_engaged(self, value: bool):
        with self._lock:
            if value:
                self._last_active = time.monotonic()
            self._set(bool(value))

    def _set(self, value: bool):
        if value and not self._engaged:
            self._epoch += 1
        if value != self._engaged:
            print(f"[spacemouse] {'ENGAGED: human control' if value else 'released: policy control'}", flush=True)
        self._engaged = value

    def state(self) -> tuple[bool, int]:
        with self._lock:
            return self._engaged, self._epoch

    @property
    def engaged(self) -> bool:
        return self.state()[0]

    def clicks(self) -> int:
        with self._lock:
            return self._clicks

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


# ---------------------------------------------------------------- controller (InterventionThread API)
class SpaceMouseIKController:
    """anchor(reading, follower14) / step(reading) -> 14-D joint targets, like RelativeIKController.
    `reading` is SpaceMice.read()."""

    def __init__(self, *, home14=None, cfg=None, **ik_kwargs):
        self.cfg = _cfg(cfg)
        home14 = None if home14 is None else np.asarray(home14, float).reshape(14)
        self.solvers = {s: MinkArmIK(None if home14 is None else home14[_slice(s)][:6], **ik_kwargs) for s in SIDES}
        self.last = np.zeros(14)
        self.targets: dict = {}
        self._grip: dict = {}
        self._t = None

    def anchor(self, reading, follower14) -> None:
        follower14 = np.asarray(follower14, float).reshape(14)
        self.last = follower14.copy()
        self.targets = {s: self.solvers[s].fk(follower14[_slice(s)][:6]) for s in SIDES}
        self._grip = {s: dict(cmd=float(np.clip(follower14[_slice(s)][6], 0.0, 1.0)), latched=False) for s in SIDES}
        self._t = time.monotonic()

    def step(self, reading) -> np.ndarray:
        if not self.targets:
            raise RuntimeError("SpaceMouseIKController.step() before anchor()")
        c = self.cfg
        now = time.monotonic()
        dt = float(np.clip(now - self._t, 0.0, 0.05))
        self._t = now
        result = self.last.copy()
        for side in SIDES:
            twist, closing = reading[side]
            pos, wxyz = self.targets[side]
            pos = pos + twist[:3] * float(c["max_lin_vel"]) * dt
            w = twist[3:] * float(c["max_rot_vel"]) * dt
            ang = float(np.linalg.norm(w))
            if ang > 1e-9:  # world(base)-frame rotation increment
                dq = np.concatenate(([np.cos(ang / 2)], w / ang * np.sin(ang / 2)))
                wxyz = _normalize(_multiply(dq, wxyz))
            sl = _slice(side)
            arm = result[sl]
            arm[:6] = self.solvers[side].solve(pos, wxyz, arm[:6])
            # anti-wind-up: keep the target within max_lead of where the IK actually got
            fpos, fwxyz = self.solvers[side].fk(arm[:6])
            lead = pos - fpos
            n = float(np.linalg.norm(lead))
            if n > c["max_lead_m"]:
                pos = fpos + lead * (c["max_lead_m"] / n)
            if quat_angle(wxyz, fwxyz) > c["max_lead_rad"]:
                wxyz = fwxyz
            self.targets[side] = (pos, wxyz)
            g = self._grip[side]
            if closing:
                g["latched"] = True
            if g["latched"]:
                goal = float(c["grip_closed"]) if closing else 0.0
                step = float(c["grip_rate"]) * dt
                g["cmd"] += float(np.clip(goal - g["cmd"], -step, step))
            arm[6] = g["cmd"]
        self.last = result
        return result.copy()


# ---------------------------------------------------------------- dry check
def _main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true")
    p.add_argument("--operator_yaw_deg", type=float, default=None)
    p.add_argument("--raw", action="store_true", help="also print the raw evdev values")
    args = p.parse_args()
    if args.list:
        for d in list_spacemice():
            print(d.path, d.name, (d.phys or "").split("/")[0])
        return
    sm = SpaceMice({"operator_yaw_deg": args.operator_yaw_deg})
    names = ["fwd(+x)", "left(+y)", "up(+z)", "roll(+x)", "pitch(+y)", "yaw(+z)"]
    print("Dry check (no robot). Push each puck away from you, left, up, and twist; the arm-frame "
          "command must match. Ctrl+C to stop.")
    try:
        while True:
            r = sm.read()
            parts, rawp = [], []
            for side in SIDES:
                tw, b0 = r[side]
                big = [f"{names[i]}{'+' if tw[i] > 0 else '-'}{abs(tw[i]):.2f}" for i in np.argsort(-np.abs(tw)) if abs(tw[i]) > 0.05][:2]
                parts.append(f"{side[0].upper()}: {' '.join(big) or '-':34s} {'GRIP' if b0 else '    '}")
                rawp.append(f"{side[0].upper()} raw {np.round(sm._raw[side]).astype(int).tolist()}")
            print(" | ".join(parts) + f" | {'HUMAN' if sm.engaged else 'policy'}"
                  + ("   [" + " ".join(rawp) + "]" if args.raw else ""), flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        sm.close()


if __name__ == "__main__":
    _main()
