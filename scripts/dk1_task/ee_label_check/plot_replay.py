"""Plot robot_replay.py runs side by side: hand path vs the demo, hand speed, and command step size,
with chunk boundaries marked.

  client_dk1/.venv/bin/python scripts/dk1_task/ee_label_check/plot_replay.py joint step cmd
  client_dk1/.venv/bin/python scripts/dk1_task/ee_label_check/plot_replay.py step_square cmd_square --out square.png
(needs matplotlib: `uv pip install --python client_dk1/.venv/bin/python matplotlib`)
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from chunk_ee import ARM, REPO, ChunkEE  # noqa: E402

COLORS = {"joint": "#6b7280", "step": "#d55e00", "cmd": "#0072b2", "chunk": "#cc79a7"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+", help="run tags, e.g. joint step cmd step_square")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--dir", default=os.path.join(REPO, "logs", "ee_label_check"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ce = ChunkEE()
    runs = {t: np.load(os.path.join(args.dir, f"ep{args.episode}_{t}.npz")) for t in args.tags}
    first = max(runs.values(), key=lambda d: len(d["q_demo"]))
    R = int(first["replan"]) if "replan" in first else 8
    demo = np.stack([ce.hand(x) for x in first["q_demo"]]) * 100  # (T, 2, 3) cm
    arm = int(np.argmax(np.ptp(demo, axis=0).sum(axis=1)))  # the arm that moves most
    side = ("left", "right")[arm]
    T = len(demo)
    t = np.arange(T) / 15.0

    fig, ax = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
    for i, xyz in enumerate("xyz"):
        ax[i].plot(t, demo[:, arm, i], color="black", lw=2.5, alpha=0.35, label="demo (measured)")
    for tag, d in runs.items():
        col = COLORS.get(tag.split("_")[0], None)
        h = np.stack([ce.hand(x) for x in d["q_meas"]]) * 100
        tr = np.arange(len(h)) / 15.0
        for i in range(3):
            ax[i].plot(tr, h[:, arm, i], color=col, lw=1.4, label=tag)
        speed = np.linalg.norm(np.diff(h[:, arm], axis=0), axis=1) * 10  # mm/step
        ax[3].plot(tr[1:], speed, color=col, lw=1.2, label=tag)
        dc = np.abs(np.diff(d["cmd"][:, ARM], axis=0)).max(axis=1)
        ax[4].plot(tr[1:], dc, color=col, lw=1.2, label=tag)
    for i, xyz in enumerate("xyz"):
        ax[i].set_ylabel(f"{side} hand {xyz} (cm)")
    ax[3].set_ylabel(f"{side} hand speed (mm/step)")
    ax[4].set_ylabel("command step, max joint (rad)")
    ax[4].set_xlabel("time (s); dotted lines = chunk boundaries (every %d steps)" % R)
    for a in ax:
        for b in np.arange(R, T, R) / 15.0:
            a.axvline(b, color="#999", lw=0.5, ls=":")
        a.grid(alpha=0.2)
    ax[0].legend(loc="upper right", fontsize=8, ncol=len(runs) + 1)
    fig.suptitle(f"demo {args.episode} open-loop replay: " + ", ".join(args.tags))
    fig.tight_layout()
    out = args.out or os.path.join(args.dir, f"ep{args.episode}_" + "_vs_".join(args.tags) + ".png")
    out = out if os.path.isabs(out) or os.path.dirname(out) else os.path.join(args.dir, out)
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
