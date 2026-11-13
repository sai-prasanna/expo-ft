"""Experimental stage 1 (no robot): does the chunk-relative EE label survive the round trip, and does it
stop errors from adding up?

  .venv/bin/python scripts/dk1_task/ee_label_check/offline.py              # 10 eps round trip, 5 eps sim
  .venv/bin/python scripts/dk1_task/ee_label_check/offline.py --rt_episodes all --sim_episodes all

1. Round trip: labels(c[t0:t0+H], q[t0]) -> IK at q[t0] -> joints, vs the recorded commands c.
2. Follower fit: per arm joint, q_{t+1} - q_t = alpha (c_t - q_t) + beta  (least squares, all demos).
3. Simulated replay with that follower, "policy" = demo labels + error, three label conventions:
     joint  absolute joint target c_t (error added in EE space, then IK)
     chunk  chunk-relative, re-anchored on the SIMULATED measured joints every `replan` steps
     step   current step-relative EE labels, re-anchored on the simulated measured joints every step
   errors: none / N(0, 2 mm) per axis per step / 1 mm per step towards -z (both arms).
   Reported: hand position error vs the demo's measured hand, per arm, final and max (cm).
"""

import argparse
import time

import numpy as np

from chunk_ee import ARM, DEFAULT_DATA, ChunkEE, load_demos, perturb_pos
from expo_ft.env.dk1_ee import MAX_LIN_DELTA, SIDES, _jslice


# ------------------------------------------------------------------------------------- 1. round trip
def round_trip(ce, demos, H, R):
    jerr, perr, rerr = [], [], []
    for ep, q, c in demos:
        for t0 in range(0, len(q), R):
            cmds = c[t0:t0 + H]
            J = ce.chunk_joints(ce.chunk_labels(cmds, q[t0]), q[t0])
            jerr.append(np.abs(J - cmds)[:, ARM].max(axis=1))
            for k in range(len(cmds)):
                perr.append(np.linalg.norm(ce.hand(J[k]) - ce.hand(cmds[k]), axis=1).max())
    jerr = np.concatenate(jerr)
    perr = np.asarray(perr)
    print(f"\n[1] Round trip, {len(demos)} episodes, H={H}, anchor every {R} steps, {len(perr)} labels")
    for name, v, unit, scale in (("max |joint - c| (rad)", jerr, "rad", 1), ("hand pos error (mm)", perr, "mm", 1e3)):
        p = np.percentile(v * scale, [50, 95, 99, 100])
        print(f"    {name:24s} median {p[0]:.4f}  p95 {p[1]:.4f}  p99 {p[2]:.4f}  max {p[3]:.4f}")
    bad = (perr > 2e-3).mean() * 100
    print(f"    labels with hand error > 2 mm: {bad:.2f}%   (pass: ~0%, max well under 5 mm)")


# ------------------------------------------------------------------------------------- 2. follower fit
def fit_follower(demos):
    alpha, beta, r2 = np.zeros(14), np.zeros(14), np.zeros(14)
    for j in ARM:
        x = np.concatenate([c[:-1, j] - q[:-1, j] for _, q, c in demos])
        y = np.concatenate([q[1:, j] - q[:-1, j] for _, q, c in demos])
        A = np.stack([x, np.ones_like(x)], 1)
        (alpha[j], beta[j]), *_ = np.linalg.lstsq(A, y, rcond=None)
        r2[j] = 1 - np.var(y - A @ np.array([alpha[j], beta[j]])) / max(np.var(y), 1e-12)
    print("\n[2] Follower fit  q_{t+1} - q_t = alpha (c_t - q_t) + beta")
    for side in SIDES:
        sl = _jslice(side)
        idx = range(sl.start, sl.start + 6)
        print(f"    {side:5s} alpha " + " ".join(f"{alpha[i]:.2f}" for i in idx)
              + "   R2 " + " ".join(f"{r2[i]:.2f}" for i in idx))
    return alpha, beta


# ------------------------------------------------------------------------------------- 3. sim
PERTS = {"none": (0.0, None), "noise 2mm": (0.002, None), "bias 1mm -z": (0.0, np.array([0, 0, -0.001]))}


def simulate(ce, q, c, mode, pert, alpha, beta, H, R, cap, rng, follower="exact"):
    """follower "fit": the fitted model alone. "exact": the fitted model plus the demo's own per-step
    residual, so it reproduces the demo exactly when given the demo's commands (only label error moves it)."""
    noise, bias = PERTS[pert]
    T = len(q)
    resid = np.zeros_like(q)
    if follower == "exact":
        resid[:-1, ARM] = q[1:, ARM] - q[:-1, ARM] - alpha[ARM] * (c[:-1, ARM] - q[:-1, ARM]) - beta[ARM]
    qs = np.zeros_like(q)
    qs[0] = q[0]
    plan, plan_t0 = None, None
    for t in range(T - 1):
        if mode == "joint":
            if noise or bias is not None:
                lab = perturb_pos(ce.chunk_labels(c[t:t + 1], c[t]), rng, noise, bias)
                cmd = ce.chunk_joints(lab, c[t])[0]
            else:
                cmd = c[t].copy()
        elif mode == "chunk":
            if t % R == 0:
                lab = perturb_pos(ce.chunk_labels(c[t:t + H], q[t]), rng, noise, bias)
                plan, plan_t0 = ce.chunk_joints(lab, qs[t]), t
            cmd = plan[t - plan_t0].copy()
        else:  # step
            lab = ce.step_label(c[t], q[t])
            for side in SIDES:
                s = _jslice(side).start
                if noise:
                    lab[s:s + 3] += rng.normal(0, noise, 3) / MAX_LIN_DELTA
                if bias is not None:
                    lab[s:s + 3] += bias / MAX_LIN_DELTA
            cmd = ce.step_joints(lab, qs[t])
        cmd[ARM] = qs[t, ARM] + np.clip(cmd[ARM] - qs[t, ARM], -cap, cap)  # env per-step joint cap
        qs[t + 1, ARM] = qs[t, ARM] + alpha[ARM] * (cmd[ARM] - qs[t, ARM]) + beta[ARM] + resid[t, ARM]
        qs[t + 1, [6, 13]] = q[t + 1, [6, 13]]
    err = np.stack([np.linalg.norm(ce.hand(qs[t]) - ce.hand(q[t]), axis=1) for t in range(T)])  # (T,2)
    return err


def sim_all(ce, demos, alpha, beta, H, R, cap, seed, follower):
    modes = ("joint", "chunk", "step")
    res = {(m, p): [] for m in modes for p in PERTS}
    for i, (ep, q, c) in enumerate(demos):
        t0 = time.time()
        for m in modes:
            for p in PERTS:
                res[(m, p)].append(simulate(ce, q, c, m, p, alpha, beta, H, R, cap, np.random.default_rng(seed + ep),
                                             follower))
        print(f"    sim episode {ep} ({len(q)} steps) {time.time() - t0:.0f}s", flush=True)
    print(f"\n[3] Simulated replay, follower={follower!r}, {len(demos)} episodes: hand error vs demo (cm), mean over episodes")
    print(f"    {'labels':6s} {'error':12s} | {'left final':>10s} {'left max':>9s} | {'right final':>11s} {'right max':>9s}")
    for m in modes:
        for p in PERTS:
            errs = res[(m, p)]
            fin = np.mean([e[-1] for e in errs], axis=0) * 100
            mx = np.mean([e.max(axis=0) for e in errs], axis=0) * 100
            print(f"    {m:6s} {p:12s} | {fin[0]:10.2f} {mx[0]:9.2f} | {fin[1]:11.2f} {mx[1]:9.2f}")
        print()
    print("    pass: chunk rows close to the joint rows, and not growing with episode length; step rows drift.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--rt_episodes", default="10", help="N (first N), comma list, or all")
    ap.add_argument("--sim_episodes", default="5", help="N (first N), comma list, or all")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--replan", type=int, default=8)
    ap.add_argument("--cap", type=float, default=0.25, help="per-step joint cap vs measured (rad), as in the env")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_rt", action="store_true")
    ap.add_argument("--skip_sim", action="store_true")
    ap.add_argument("--followers", default="exact,fit", help="exact: model + demo residual; fit: model alone")
    args = ap.parse_args()

    all_demos = load_demos(args.data)
    ids = [ep for ep, *_ in all_demos]
    pick = lambda s: all_demos if s == "all" else (
        [d for d in all_demos if d[0] in {int(x) for x in s.split(",")}] if "," in s else
        [d for d in all_demos if d[0] in set(ids[:int(s)])])
    print(f"{len(all_demos)} demos in {args.data}")
    ce = ChunkEE()

    if not args.skip_rt:
        t0 = time.time()
        round_trip(ce, pick(args.rt_episodes), args.horizon, args.replan)
        print(f"    ({time.time() - t0:.0f}s)")
    alpha, beta = fit_follower(all_demos)
    if not args.skip_sim:
        for fol in args.followers.split(","):
            sim_all(ce, pick(args.sim_episodes), alpha, beta, args.horizon, args.replan, args.cap, args.seed, fol)


if __name__ == "__main__":
    main()
