"""Scene-reset guide: a live head-camera window with markers for where to put the cube and the bowl.

DK1Env.reset() shows it while waiting for "reset the scene" (task config `layout_guide`). Layouts
are sampled with `seed` from the demo start positions in the layout file (configs/task/layouts/):
each object at a random demo start of that object plus ~1.5 cm jitter, kept apart and away from
the gripper home spots. mode "eval" and mode "train" are separate streams: every eval run walks the
same E1, E2, ...; training gets T1, T2, ... and resumes after the last row of its log on restart.

A marker turns from orange to green once the detected object is within `tol_px`. Each episode's
layout and outcome are appended to a CSV in logs/. Window keys: n / p = next / previous layout.
"""

from __future__ import annotations

import csv
import json
import os
import time

import cv2
import numpy as np

WINDOW = "scene reset guide"
OBJECTS = ("cube", "bowl")


def detect(rgb: np.ndarray) -> dict:
    """Largest blue blob (cube) and green blob (bowl) in an RGB frame -> {name: (x, y) | None}."""
    a = rgb.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    masks = {"cube": (b > r + 40) & (b > g + 30), "bowl": (g > b + 50) & (g > r + 15) & (g > 120)}
    min_area = {"cube": 15, "bowl": 60}
    scale = rgb.shape[0] * rgb.shape[1] / (240 * 320)
    out = {}
    for name, m in masks.items():
        n, _, stats, cents = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) if n > 1 else None
        out[name] = tuple(cents[k]) if k and stats[k, cv2.CC_STAT_AREA] >= min_area[name] * scale else None
    return out


def sample_layout(rng: np.random.Generator, doc: dict, jitter_px=8.0, min_sep_px=40.0, gripper_clear_px=35.0):
    demo = {o: np.asarray(doc["demo_positions"][o], float) for o in OBJECTS}
    lo = np.minimum(demo["cube"].min(0), demo["bowl"].min(0))
    hi = np.maximum(demo["cube"].max(0), demo["bowl"].max(0))
    grips = np.asarray(doc["gripper_home"], float)
    while True:
        pos = {o: demo[o][rng.integers(len(demo[o]))] + rng.normal(0.0, jitter_px, 2) for o in OBJECTS}
        if (np.linalg.norm(pos["cube"] - pos["bowl"]) >= min_sep_px
                and all(np.all((lo <= p) & (p <= hi)) for p in pos.values())
                and all(np.linalg.norm(grips - p, axis=1).min() >= gripper_clear_px for p in pos.values())):
            return {o: [round(float(v), 1) for v in pos[o]] for o in OBJECTS}


class LayoutGuide:
    def __init__(self, path: str, mode: str = "eval", seed: int = 0, tol_px: float = 6.0, show: bool = True):
        assert mode in ("eval", "train"), mode
        with open(path) as f:
            self.doc = json.load(f)
        self.mode, self.seed, self.tol, self.show = mode, int(seed), float(tol_px), show
        stem = os.path.splitext(os.path.basename(path))[0]
        if mode == "train":  # one log per seed; a restart continues the stream after its last row
            self.log_path = os.path.join("logs", f"layouts_train_{stem}_seed{seed}.csv")
            with open(self.log_path) if os.path.exists(self.log_path) else open(os.devnull) as f:
                self.index = max(0, sum(1 for _ in f) - 1)
        else:
            self.log_path = os.path.join("logs", f"layouts_eval_{stem}_seed{seed}_{time.strftime('%Y%m%d-%H%M%S')}.csv")
            self.index = 0
        self.current = None
        self._win = None
        print(f"[layout] {mode} layouts, seed {seed}, from #{self.index + 1}; log -> {self.log_path}", flush=True)

    def layout(self) -> dict:
        stream = 1 if self.mode == "eval" else 0
        lay = sample_layout(np.random.default_rng([self.seed, stream, self.index]), self.doc)
        return {"id": f"{'E' if self.mode == 'eval' else 'T'}{self.index + 1}", **lay}

    def render(self, rgb: np.ndarray, lay: dict) -> np.ndarray:
        h, w = rgb.shape[:2]
        sx, sy = w / self.doc["ref_size"][0], h / self.doc["ref_size"][1]
        img = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        found = detect(rgb)
        text = [f"{lay['id']}   n/p = other layout, Enter / pedal = start"]
        for name in OBJECTS:
            t = (int(lay[name][0] * sx), int(lay[name][1] * sy))
            cur = found[name]
            dist = None if cur is None else np.hypot(cur[0] - t[0], cur[1] - t[1]) / sx
            col = (0, 220, 0) if dist is not None and dist <= self.tol else (0, 165, 255)
            if name == "cube":
                cv2.rectangle(img, (t[0] - 9, t[1] - 9), (t[0] + 9, t[1] + 9), col, 2)
            else:
                cv2.circle(img, t, 20, col, 2)
            cv2.putText(img, name, (t[0] + 22, t[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            if cur is not None and col != (0, 220, 0):
                cv2.arrowedLine(img, (int(cur[0]), int(cur[1])), t, col, 2, tipLength=0.2)
            text.append(f"{name}: " + ("not seen" if dist is None else f"{dist:.0f}px off"))
        for i, s in enumerate(text):
            for c, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                cv2.putText(img, s, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, th, cv2.LINE_AA)
        return img

    # DK1Env.reset(): begin() -> tick(frame) while waiting -> end(); episode over -> record(outcome)
    def begin(self):
        lay = self.layout()
        print(f"[layout] Place {lay['id']}: cube {lay['cube']}, bowl {lay['bowl']}.", flush=True)

    def tick(self, rgb: np.ndarray):
        if not self.show:
            return
        if self._win is None:
            self._win = _TkWindow(WINDOW, on_key=self._key)
        self._win.show(cv2.cvtColor(self.render(rgb, self.layout()), cv2.COLOR_BGR2RGB))

    def _key(self, ch: str):
        if ch in ("n", "p"):
            self.index = max(0, self.index + (1 if ch == "n" else -1))
            self.begin()

    def end(self):
        self.current = self.layout()
        if self._win is not None:
            self._win.close()
            self._win = None

    def record(self, outcome: str, steps: int):
        lay = self.current
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        new = not os.path.exists(self.log_path)
        with open(self.log_path, "a", newline="") as f:
            wr = csv.writer(f)
            if new:
                wr.writerow(["time", "layout", "cube_x", "cube_y", "bowl_x", "bowl_y", "outcome", "steps"])
            wr.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), lay["id"], *lay["cube"], *lay["bowl"], outcome, steps])
        print(f"[layout] {lay['id']}: {outcome.upper()}", flush=True)
        self.index += 1


class _TkWindow:
    """Minimal image window (the client's OpenCV is headless). Call from the main thread."""

    def __init__(self, title: str, on_key=None):
        import tkinter as tk

        self.root = tk.Tk()
        self.root.title(title)
        self.label = tk.Label(self.root)
        self.label.pack()
        if on_key is not None:
            self.root.bind("<Key>", lambda e: on_key(e.char))
        self.root.attributes("-topmost", True)

    def show(self, rgb: np.ndarray):
        from PIL import Image, ImageTk

        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))  # keep a reference
        self.label.configure(image=self._photo)
        self.root.update()

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass
