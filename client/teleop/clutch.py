"""Intervention clutch: a key toggle that hands control between the policy and the leader arms.

Engaging re-anchors the relative IK; disengaging hands control back to the policy.

Backends (task config `clutch_backend`):

- "terminal" (default, works everywhere incl. Wayland/SSH): run
      python -m client.teleop.clutch
  in a SECOND terminal. Every SPACE/ENTER there toggles the clutch (q quits). Toggles are
  sent as UDP datagrams to 127.0.0.1:`clutch_port`. The main terminal stays free for the
  manual success detector, which reads lines ("1"/"2") from /dev/tty.
- "pynput": global key listener (X11 only; Wayland terminals are invisible to it). Default
  key is F8. Never use the digits 1/2 (taken by the success detector).
- "manual": no input; toggle from code via `set_engaged()` (mock tests).
"""

from __future__ import annotations

import socket
import sys
import threading

DEFAULT_PORT = 8765


class Clutch:
    def __init__(self, backend: str = "terminal", port: int = DEFAULT_PORT, key: str = "f8"):
        self.backend = backend
        self._engaged = False
        self._epoch = 0  # incremented on every engage; lets the controller detect re-anchors
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._listener = None
        if backend == "terminal":
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.bind(("127.0.0.1", int(port)))
            self._sock.settimeout(0.2)
            self._thread = threading.Thread(target=self._udp_loop, daemon=True, name="clutch-udp")
            self._thread.start()
            print(f"[clutch] listening on udp://127.0.0.1:{port}; run `python -m client.teleop.clutch` "
                  f"in another terminal and press SPACE to take over / hand back.")
        elif backend == "pynput":
            from pynput import keyboard

            target = getattr(keyboard.Key, key, None) or keyboard.KeyCode.from_char(key)
            if str(key) in ("1", "2"):
                raise ValueError("Clutch key must not be 1/2 (used by the manual success detector).")

            def on_press(k):
                if k == target:
                    self.toggle()

            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.start()
            print(f"[clutch] pynput: press {key} to take over / hand back.")
        elif backend != "manual":
            raise ValueError(f"Unknown clutch backend {backend!r}")

    def _udp_loop(self):
        while not self._stop.is_set():
            try:
                msg, _ = self._sock.recvfrom(64)
            except socket.timeout:
                continue
            except OSError:
                break
            cmd = msg.decode(errors="ignore").strip()
            if cmd == "toggle":
                self.toggle()
            elif cmd == "on":
                self.set_engaged(True)
            elif cmd == "off":
                self.set_engaged(False)

    def toggle(self):
        with self._lock:
            self._set(not self._engaged)

    def set_engaged(self, value: bool):
        with self._lock:
            self._set(bool(value))

    def _set(self, value: bool):
        if value and not self._engaged:
            self._epoch += 1
        if value != self._engaged:
            print(f"[clutch] {'ENGAGED: human control' if value else 'released: policy control'}", flush=True)
        self._engaged = value

    def state(self) -> tuple[bool, int]:
        with self._lock:
            return self._engaged, self._epoch

    @property
    def engaged(self) -> bool:
        return self.state()[0]

    def close(self):
        self._stop.set()
        if self._listener is not None:
            self._listener.stop()
        if self.backend == "terminal":
            try:
                self._sock.close()
            except OSError:
                pass


def _terminal_main(port: int = DEFAULT_PORT):
    """Second-terminal key sender: SPACE/ENTER toggles, o = on, f = off, q quits."""
    import termios
    import tty

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = ("127.0.0.1", int(port))
    print(f"Clutch -> udp://127.0.0.1:{port}. SPACE/ENTER: toggle | o: engage | f: release | q: quit")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in (" ", "\n", "\r"):
                sock.sendto(b"toggle", addr)
                print("toggle", flush=True)
            elif ch == "o":
                sock.sendto(b"on", addr)
                print("engage", flush=True)
            elif ch == "f":
                sock.sendto(b"off", addr)
                print("release", flush=True)
            elif ch in ("q", "\x03", "\x04"):
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    _terminal_main(p.parse_args().port)
