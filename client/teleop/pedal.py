"""Foot pedal: a spare mouse on the floor that hands control between the policy and the leader arms.

HOLD any button = human control (the relative IK anchors on every press, so: release, re-center
the leader, press again). Release = policy control. A short CLICK (press + release) also answers
the env's "reset the scene, then press Enter" prompt (`clicks()` counts releases).

The mouse is matched by name (task config `intervention.pedal_name`, substring of the evdev name).
All evdev interfaces of the same USB receiver are read and grabbed (some buttons arrive on the
receiver's keyboard interface, e.g. as the Super/"Windows" key), so every button counts and none
reaches the desktop. Unplugging it / read errors release to the policy; it reconnects when it
comes back. Needs read access to the devices: scripts/dk1_task/70-dk1-footpedal.rules.

`Pedal(None)` reads no device; press/release from code via `set_engaged()` (mock tests).
`Pedal(name, clicker=True)`: a plain start/stop button (data collection): only `clicks()` matters,
no takeover / human-control messages.
"""

from __future__ import annotations

import select
import threading

DEFAULT_PEDAL_NAME = "2.4GHz Transceiver v9.0 Mouse"  # Microsoft receiver mouse, 045e:07a5


def _usb_port(dev) -> str:
    """'usb-0000:00:14.0-9.1/input1' -> 'usb-0000:00:14.0-9.1' (shared by all interfaces of the receiver)."""
    return (dev.phys or "").split("/")[0]


def find_pedal(name: str) -> list:
    """[the evdev mouse whose name contains `name` (case-insensitive) and has buttons, then every
    other readable evdev device on the same USB receiver]."""
    import evdev

    devs = []
    for path in evdev.list_devices():
        try:
            devs.append(evdev.InputDevice(path))
        except OSError:
            continue
    mouse = next((d for d in devs if name.lower() in d.name.lower()
                  and evdev.ecodes.BTN_LEFT in d.capabilities().get(evdev.ecodes.EV_KEY, [])), None)
    if mouse is None:
        for d in devs:
            d.close()
        raise FileNotFoundError(f"No readable mouse named like {name!r} (unplugged, or no access to "
                                f"/dev/input/event*: install scripts/dk1_task/70-dk1-footpedal.rules)")
    port = _usb_port(mouse)
    found = [mouse]
    for d in devs:
        if d is mouse:
            continue
        if port and _usb_port(d) == port and evdev.ecodes.EV_KEY in d.capabilities():
            found.append(d)
        else:
            d.close()
    return found


class Pedal:
    def __init__(self, name: str | None = DEFAULT_PEDAL_NAME, clicker: bool = False):
        self.clicker = bool(clicker)
        self._engaged = False
        self._epoch = 0  # incremented on every press; lets the controller detect re-anchors
        self._clicks = 0  # incremented on every release (from the device only)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._name = name
        self._devs: list = []
        if name is None:
            return
        self._devs = find_pedal(name)  # fail loudly at startup if it is missing
        self._thread = threading.Thread(target=self._loop, daemon=True, name="foot-pedal")
        self._thread.start()
        extra = f" (+{len(self._devs) - 1} sibling interfaces)" if len(self._devs) > 1 else ""
        if self.clicker:
            print(f"[pedal] {self._devs[0].name} ({self._devs[0].path}){extra}: click = start / stop recording.")
        else:
            print(f"[pedal] {self._devs[0].name} ({self._devs[0].path}){extra}: HOLD to take over, "
                  f"release to hand back; click = Enter at the scene-reset prompt.")

    def _close_devs(self, ungrab: bool):
        for d in self._devs:
            try:
                if ungrab:
                    d.ungrab()
            except OSError:
                pass
            try:
                d.close()
            except OSError:
                pass
        self._devs = []

    def _loop(self):
        import evdev

        while not self._stop.is_set():
            if not self._devs:
                try:
                    self._devs = find_pedal(self._name)
                    print(f"[pedal] reconnected: {self._devs[0].path}", flush=True)
                except OSError:  # FileNotFoundError included
                    self._stop.wait(1.0)
                    continue
            try:
                by_fd = {d.fd: d for d in self._devs}
                held = set()
                for d in self._devs:
                    try:
                        d.grab()  # the pedal never reaches the desktop
                    except OSError:
                        pass
                    held |= {(d.fd, k) for k in d.active_keys()}  # buttons already down on (re)connect
                self.set_engaged(bool(held))
                while not self._stop.is_set():
                    ready = select.select(list(by_fd), [], [], 0.2)[0]
                    for fd in ready:
                        for ev in by_fd[fd].read():
                            if ev.type != evdev.ecodes.EV_KEY:
                                continue
                            if ev.value == 1:
                                held.add((fd, ev.code))
                            elif ev.value == 0 and (fd, ev.code) in held:
                                held.discard((fd, ev.code))
                                if not held:
                                    with self._lock:
                                        self._clicks += 1
                            self.set_engaged(bool(held))
            except OSError as e:  # unplugged / receiver lost
                if not self._stop.is_set():
                    print(f"\n[pedal] !!! foot pedal lost ({e!r})"
                          f"{'' if self.clicker else '; policy control'} !!!\n", flush=True)
                self.set_engaged(False)
                self._close_devs(ungrab=False)
        self._close_devs(ungrab=True)

    def set_engaged(self, value: bool):
        with self._lock:
            self._set(bool(value))

    def _set(self, value: bool):
        if value and not self._engaged:
            self._epoch += 1
        if value != self._engaged and not self.clicker:
            print(f"[pedal] {'ENGAGED: human control' if value else 'released: policy control'}", flush=True)
        self._engaged = value

    def state(self) -> tuple[bool, int]:
        with self._lock:
            return self._engaged, self._epoch

    def clicks(self) -> int:
        """Number of completed press+release cycles so far (compare before/after to detect a click)."""
        with self._lock:
            return self._clicks

    @property
    def engaged(self) -> bool:
        return self.state()[0]

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
