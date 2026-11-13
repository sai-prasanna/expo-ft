"""Experimental: label conventions for the DK1 EE action space (not used by training yet).

  step   (current, expo_ft/env/dk1_ee.py): a_t = EE(c_t) (-) EE(q_t), applied at EVERY step to the
         measured hand: target_t = EE(q_t) (+) a_t.
  chunk  (proposed): for a chunk starting at t0, a_{t0+k} = EE(c_{t0+k}) (-) EE(q_t0), k = 0..H-1,
         all applied to the measured hand at the chunk start: target = EE(q_t0) (+) a_{t0+k}.
  joint  (reference): send the recorded joint command c_t as is.

Per arm, in that arm's base frame: [dpos (m, 3), rotvec of R_target R_anchor^-1 (rad, 3), gripper].
"""

from __future__ import annotations

import glob
import os
import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from expo_ft.env.dk1_ee import DK1EE, SIDES, _jslice, _wxyz  # noqa: E402

ARM = np.r_[0:6, 7:13]
DEFAULT_DATA = os.path.join(REPO, "data", "dk1_pnp_bowl_new", "success")


def load_demos(data_dir=DEFAULT_DATA, episodes=None):
    """[(ep_id, q (T,14) measured, c (T,14) executed command)]"""
    paths = sorted(glob.glob(os.path.join(data_dir, "*", "traj.hdf5")),
                   key=lambda p: int(os.path.basename(os.path.dirname(p))))
    out = []
    for p in paths:
        ep = int(os.path.basename(os.path.dirname(p)))
        if episodes is not None and ep not in episodes:
            continue
        with h5py.File(p) as f:
            out.append((ep, f["saved_observation/state"][:].astype(np.float64),
                        f["action/action14"][:].astype(np.float64)))
    return out


class ChunkEE:
    def __init__(self):
        self.ee = DK1EE()
        self.ee._ik = self.ee._make_ik()

    # ---------------------------------------------------------------- poses
    def hand(self, joints14):
        """(2,3) hand positions [left, right]."""
        j = np.asarray(joints14, float).reshape(14)
        return np.stack([self.ee.pose(j[_jslice(s)][:6])[0] for s in SIDES])

    # ---------------------------------------------------------------- chunk-relative
    def chunk_labels(self, cmds, anchor14):
        """cmds (H,14) -> labels (H,14) relative to the anchor (measured joints at the chunk start)."""
        cmds = np.asarray(cmds, float).reshape(-1, 14)
        out = np.zeros_like(cmds)
        for side in SIDES:
            sl = _jslice(side)
            pa, ra = self.ee.pose(np.asarray(anchor14, float)[sl][:6])
            for k, c in enumerate(cmds):
                pt, rt = self.ee.pose(c[sl][:6])
                out[k, sl] = np.concatenate([pt - pa, (rt * ra.inv()).as_rotvec(), [c[sl][6]]])
        return out

    def chunk_joints(self, labels, anchor14):
        """labels (H,14) at the measured anchor -> joint targets (H,14). IK seeded with the anchor for
        k=0, then with the previous solution (continuous within the chunk)."""
        labels = np.asarray(labels, float).reshape(-1, 14)
        anchor14 = np.asarray(anchor14, float).reshape(14)
        out = np.repeat(anchor14[None], len(labels), axis=0)
        for side in SIDES:
            sl = _jslice(side)
            pa, ra = self.ee.pose(anchor14[sl][:6])
            seed = anchor14[sl][:6]
            for k, a in enumerate(labels):
                a = a[sl]
                pt = pa + a[:3]
                rt = Rotation.from_rotvec(a[3:6]) * ra
                seed = self.ee._ik[side].solve(pt, _wxyz(rt), seed=seed)
                out[k, sl.start:sl.start + 6] = seed
                out[k, sl.start + 6] = a[6]
        return out

    # ---------------------------------------------------------------- step-relative (current)
    def step_label(self, cmd14, current14):
        return self.ee.action(cmd14, current14).astype(np.float64)

    def step_joints(self, label14, current14):
        return self.ee.joints(label14, current14)


def perturb_pos(labels, rng, noise_m=0.0, bias_m=None):
    """Add per-label position noise (N(0, noise_m) per axis) and a constant bias (3,) in metres to both
    arms. Labels are (H,14) chunk labels (metres)."""
    labels = np.array(labels, float, copy=True).reshape(-1, 14)
    for side in SIDES:
        s = _jslice(side).start
        if noise_m:
            labels[:, s:s + 3] += rng.normal(0, noise_m, (len(labels), 3))
        if bias_m is not None:
            labels[:, s:s + 3] += np.asarray(bias_m, float)
    return labels
