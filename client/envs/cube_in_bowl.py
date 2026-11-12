"""Success detector for pick-into-bowl from the head camera + gripper state.

Success = the blue cube's centre lies within `frac` bowl radii of the green bowl's centre in the
head image AND both grippers are open (cube released), for `k` consecutive steps.

Checked offline on the 50 dk1_pnp_bowl_new demos (2026-11-12): fired in 50/50, never before the
release, 9-63 steps after it. The release condition matters: in 14 demos the held cube is "in the
bowl" in the image while it is carried over it. Not yet checked on failures (cube next to / behind
the bowl in the image). At the step limit, finalize() also accepts a success held >= 3 steps.
"""

from __future__ import annotations

import cv2
import numpy as np

# At the policy's 240x320 head image; same colour rules as client/teleop/layout_guide.py.
MIN_AREA = {"cube": 15, "bowl": 60}


def blobs(rgb: np.ndarray) -> dict:
    """{"cube"/"bowl": None | {"c": (x, y), "r": radius_px}} for the largest blue / green blob."""
    a = rgb.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    masks = {"cube": (b > r + 40) & (b > g + 30), "bowl": (g > b + 50) & (g > r + 15) & (g > 120)}
    out = {}
    for name, m in masks.items():
        n, _, st, ce = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])) if n > 1 else None
        if k is None or st[k, cv2.CC_STAT_AREA] < MIN_AREA[name]:
            out[name] = None
        else:
            out[name] = {"c": ce[k], "r": 0.5 * max(st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT])}
    return out


class CubeInBowlDetector:
    def __init__(self, frac: float = 0.8, open_thresh: float = 0.3, k: int = 10):
        self.frac, self.open_thresh, self.k = float(frac), float(open_thresh), int(k)
        self.reset()

    def reset(self):
        self.run = 0
        self.fired_step = None
        self.step = 0

    def update(self, head_rgb: np.ndarray, state14: np.ndarray) -> bool:
        """One step (head image at 240x320, state with grippers at 6 and 13). True once it fires."""
        f = blobs(head_rgb)
        in_bowl = (f["cube"] is not None and f["bowl"] is not None
                   and np.linalg.norm(f["cube"]["c"] - f["bowl"]["c"]) <= self.frac * f["bowl"]["r"])
        released = state14[6] < self.open_thresh and state14[13] < self.open_thresh
        self.run = self.run + 1 if (in_bowl and released) else 0
        if self.run >= self.k and self.fired_step is None:
            self.fired_step = self.step
        self.step += 1
        return self.fired_step is not None

    def finalize(self, min_run: int = 3) -> bool:
        """At the step limit the k-step window may be cut short (a drop in the last steps): count
        a success that has held for >= min_run steps. Idempotent. -> success."""
        if self.fired_step is None and self.run >= min_run:
            self.fired_step = self.step - 1
        return self.fired_step is not None
