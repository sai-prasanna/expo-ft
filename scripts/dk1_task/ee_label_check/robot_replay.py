"""Experimental (ROBOT): replay a demo open loop (no policy) with different EE labels, with and without
EXPO-style edits, and record the cameras, so the label conventions can be compared on the hardware.

  client_dk1/.venv/bin/python scripts/dk1_task/ee_label_check/robot_replay.py --episode 0 --modes joint,step,cmd
  ... --modes step,cmd --edit square        # chunk-constant edit (what a smooth edit actor would do)
  ... --modes step,cmd --edit noise         # per-step independent edit (the untrained edit actor)
  ... --modes joint,step,cmd --interp_hz 0  # different tracking lag than when the demos were recorded

Modes (all go through DK1Env.step: joint limits, per-step joint cap, EE box, command ramp):
  joint  recorded command c_t as is (reference)
  step   CURRENT EE action space: a_t = EE(c_t) (-) EE(q_t) (demo's measured joints), applied at the
         robot's measured joints every step (expo_ft/env/dk1_ee.py today)
  cmd    PROPOSED: a_t = EE(c_t) (-) EE(c_{t-1}), applied to the last command sent (c_{-1} = q_0)
  chunk  chunk-relative to the measured joints at each chunk start (rejected; shown jerky 2026-11-13)

Edits (step / cmd only; position, both arms, horizontal, fixed random direction per arm, seed 0):
  square  every other chunk of `replan` steps carries an offset: step mode adds edit_mm to every label
          of that chunk (the target jumps by edit_mm at each chunk start); cmd mode adds edit_mm/replan
          per step (+ in edited chunks, - in the next one), the same 0..edit_mm offset as a ramp
  noise   N(0, noise_mm) per step and axis added to every label

Before each run the arms move slowly to the demo's first state; press Enter to start each run.
Clear the table (the replay closes the gripper where the cube was). Hand on the e-stop.
Saves logs/ee_label_check/ep{E}_{tag}.npz + .mp4 (head | left wrist | right wrist) and prints:
hand error vs demo, command step size at chunk boundaries vs inside chunks, hand acceleration
(jerkiness of the measured motion), and the lag between command and measured joints.
"""

import argparse
import importlib
import os
import time

import numpy as np

from chunk_ee import ARM, DEFAULT_DATA, REPO, ChunkEE, load_demos
from expo_ft.env.dk1_ee import MAX_LIN_DELTA

POS = (np.r_[0:2], np.r_[7:9])  # x, y label dims of the left / right arm


def make_edits(kind, T, R, edit_mm, noise_mm, seed=0):
    """(T, 14) label offsets in metres for step mode and for cmd mode (the same physical offset)."""
    step_e, cmd_e = np.zeros((T, 14)), np.zeros((T, 14))
    if kind == "none":
        return step_e, cmd_e
    rng = np.random.default_rng(seed)
    for dims in POS:
        if kind == "square":
            u = rng.normal(size=2)
            u *= edit_mm / 1000 / np.linalg.norm(u)
            for t in range(T):
                on = (t // R) % 2 == 0
                step_e[t, dims] = u if on else 0.0
                cmd_e[t, dims] = (u if on else -u) / R
        else:
            n = rng.normal(0, noise_mm / 1000, (T, 2))
            step_e[:, dims] = n
            cmd_e[:, dims] = n
    return step_e, cmd_e


def precheck(ce, q, c, H, R, tol_m=0.003):
    """Offline, no robot: each label convention reproduces the demo commands when the robot tracks
    like the demo (step: at the demo's measured joints; cmd: integrated from q_0; chunk: per chunk)."""
    worst = {"chunk": 0.0, "cmd": 0.0, "step": 0.0}
    for t0 in range(0, len(q), R):
        J = ce.chunk_joints(ce.chunk_labels(c[t0:t0 + H], q[t0]), q[t0])
        for k in range(len(J)):
            worst["chunk"] = max(worst["chunk"], np.linalg.norm(ce.hand(J[k]) - ce.hand(c[t0 + k]), axis=1).max())
    prev = q[0]
    for t in range(len(q)):
        j = ce.step_joints(ce.step_label(c[t], c[t - 1] if t else q[0]), prev)
        worst["cmd"] = max(worst["cmd"], np.linalg.norm(ce.hand(j) - ce.hand(c[t]), axis=1).max())
        j2 = ce.step_joints(ce.step_label(c[t], q[t]), q[t])
        worst["step"] = max(worst["step"], np.linalg.norm(ce.hand(j2) - ce.hand(c[t]), axis=1).max())
        prev = j
    print("[precheck] round trip worst hand error (mm): " + ", ".join(f"{k} {v * 1000:.2f}" for k, v in worst.items()))
    if max(worst.values()) > tol_m:
        raise SystemExit(f"round trip error > {tol_m * 1000:.0f} mm: not sending this to the robot")


def run(env, ce, q, c, mode, H, R, hz, T, edits, record):
    meas, cmds = np.zeros((T, 14)), np.zeros((T, 14))
    step_e, cmd_e = edits
    plan, plan_t0 = None, 0
    period = 1.0 / hz
    late = 0
    for t in range(T):
        t_start = time.perf_counter()
        cur = np.asarray(env.get_observation()["state"] if record else env.get_state(), dtype=np.float64)
        meas[t] = cur
        if mode == "joint":
            cmd = c[t]
        elif mode == "chunk":
            if t % R == 0:
                plan, plan_t0 = ce.chunk_joints(ce.chunk_labels(c[t:t + H], q[t]), cur), t
            cmd = plan[t - plan_t0]
        elif mode == "step":
            a = ce.step_label(c[t], q[t]) + step_e[t] / MAX_LIN_DELTA
            cmd = ce.step_joints(a, cur)
        else:  # cmd
            a = ce.step_label(c[t], c[t - 1] if t else q[0]) + cmd_e[t] / MAX_LIN_DELTA
            cmd = ce.step_joints(a, cmds[t - 1] if t else cur)
        cmds[t] = env.step(cmd)["executed_action"]
        dt = time.perf_counter() - t_start
        late += dt > period
        time.sleep(max(0.0, period - dt))
    if late:
        print(f"  [{mode}] {late}/{T} steps over the {period * 1000:.0f} ms budget")
    return meas, cmds


def report(ce, tag, meas, cmds, q, R):
    T = len(meas)
    hm = np.stack([ce.hand(m) for m in meas])  # (T, 2, 3)
    err = np.linalg.norm(hm - np.stack([ce.hand(x) for x in q[:T]]), axis=2) * 100  # cm
    thirds = np.array_split(np.arange(T), 3)
    line = []
    for i, side in enumerate(("left", "right")):
        m = [err[idx, i].mean() for idx in thirds]
        line.append(f"{side}: thirds {m[0]:.2f}/{m[1]:.2f}/{m[2]:.2f}  max {err[:, i].max():.2f}  final {err[-1, i]:.2f}")
    print(f"  [{tag}] hand error vs demo (cm)  " + "   ".join(line))
    bi = np.arange(R - 1, T - 1, R)
    dc = np.abs(np.diff(cmds[:, ARM], axis=0)).max(axis=1)
    print(f"          command step (rad): chunk boundaries mean {dc[bi].mean():.3f} max {dc[bi].max():.3f} | "
          f"inside chunks mean {np.delete(dc, bi).mean():.3f} max {np.delete(dc, bi).max():.3f}")
    acc = np.linalg.norm(np.diff(hm, n=2, axis=0), axis=2).max(axis=1) * 1000  # mm/step^2
    print(f"          hand acceleration (mm/step^2): rms {np.sqrt((acc ** 2).mean()):.2f}  p95 {np.percentile(acc, 95):.2f}")
    lag_err = [np.abs(meas[L:, ARM] - cmds[:T - L, ARM]).mean() for L in range(8)]
    L = int(np.argmin(lag_err))
    print(f"          lag: measured matches the command {L} steps later ({L * 1000 / 15:.0f} ms)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_config", default="configs.task.dk1_pick")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--modes", default="joint,step,cmd")
    ap.add_argument("--edit", default="none", choices=("none", "square", "noise"))
    ap.add_argument("--edit_mm", type=float, default=10.0, help="square edit offset (mm)")
    ap.add_argument("--noise_mm", type=float, default=1.0, help="per-step edit noise std per axis (mm)")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--replan", type=int, default=8)
    ap.add_argument("--max_joint_delta", type=float, default=0.25,
                    help="env per-step joint cap (rad). Demos were recorded with 0.25; tighter clips the replay")
    ap.add_argument("--interp_hz", type=float, default=None,
                    help="command ramp thread rate (task config: 100). 0 = off: each target goes straight to the motors")
    ap.add_argument("--steps", type=int, default=150, help="replay only the first N steps (0 = all)")
    ap.add_argument("--no_video", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO, "logs", "ee_label_check"))
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    assert all(m in ("joint", "chunk", "step", "cmd") for m in modes), modes
    assert args.edit == "none" or all(m in ("step", "cmd") for m in modes), "edits apply to step / cmd only"

    demos = load_demos(args.data, episodes={args.episode})
    if not demos:
        raise SystemExit(f"episode {args.episode} not found in {args.data}")
    _, q, c = demos[0]
    T = min(len(q), args.steps or len(q))
    ce = ChunkEE()
    precheck(ce, q, c, args.horizon, args.replan)
    edits = make_edits(args.edit, T, args.replan, args.edit_mm, args.noise_mm)

    cfg = importlib.import_module(args.task_config).get_config()
    kw = dict(cfg)
    kw.update(video_dir="", env_usage="collect", wait_for_scene_reset=False,
              max_joint_delta_per_step=args.max_joint_delta)
    if args.interp_hz is not None:
        kw["command_interp_hz"] = args.interp_hz
    print(f"[config] command_interp_hz = {kw.get('command_interp_hz')}  edit = {args.edit}")
    kw["success_detector"] = {"type": ""}
    kw["layout_guide"] = {"path": ""}
    env = cfg.env(**kw)
    hz = float(cfg.control_hz)
    os.makedirs(args.out, exist_ok=True)
    rec = None if args.no_video else env._recorder
    suffix = "" if args.edit == "none" else f"_{args.edit}"
    suffix += "" if args.interp_hz is None else f"_interp{args.interp_hz:g}"
    try:
        for mode in modes:
            tag = f"{mode}{suffix}"
            print(f"\n[{tag}] moving slowly to demo {args.episode}'s first state...", flush=True)
            env._move_to(q[0].copy())
            input(f"[{tag}] at start. Clear the table, hand on the e-stop, Enter to replay {T} steps "
                  f"({T / hz:.1f} s)...")
            if rec is not None:
                rec.start(args.out, f"ep{args.episode}_{tag}")
            meas, cmds = run(env, ce, q, c, mode, args.horizon, args.replan, hz, T, edits, rec is not None)
            if rec is not None:
                rec.finish("replay")
            report(ce, tag, meas, cmds, q, args.replan)
            np.savez(os.path.join(args.out, f"ep{args.episode}_{tag}.npz"), q_meas=meas, cmd=cmds,
                     q_demo=q[:T], c_demo=c[:T], edit_step=edits[0], edit_cmd=edits[1], replan=args.replan)
        print(f"\nsaved to {args.out}")
    finally:
        env.home()
        env.close()


if __name__ == "__main__":
    main()
