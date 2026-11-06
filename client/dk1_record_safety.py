"""Record the DK1 Cartesian safety limits on the robot and save them to dk1_hardware.json.

    source client_dk1/.venv/bin/activate
    python -m client.dk1_record_safety              # guided recording, ~3 min, teleop with the leaders
    python -m client.dk1_record_safety --check_only # no hardware: replay the demos through the saved limits

Teleop as in demo collection (followers mirror the leader joints, safety limits off). Three steps:

  1. Table floor: touch the table with each gripper's fingertips at 3 spots (Enter at each).
     Floor = highest tool0 z of the touches + `floor_margin`, per arm (absorbs table tilt).
  2. Arm-to-arm transform: touch the two fingertips together at 4 spread-out spots, at different
     heights (Enter at each). A rigid fit gives right base -> left base; the residual is printed.
  3. Workspace: sweep both arms over everything they may reach during the task (Enter when done).
     Box = sweep range + `box_margin` in x/y and on top; floor from step 1. The demos' tool0 range
     is added (never below the floor), so recorded demo motion is always allowed.

Then the demos are replayed through the final limits (state = current, action14 = command) and the
held-step count is printed; it should be 0. Writes ee_bounds, right_base_in_left and
min_arm_distance into configs/task/dk1_hardware.json (previous file kept as .bak).
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np
from absl import app, flags

from client import collect_data_dk1 as cdk  # defines --task_config, --sync_max_delta, --sync_tolerance
from client.envs.dk1_safety import ARM_JOINTS, SIDES, DK1Points, DK1Safety, check_demos, fit_rigid
from configs.task.dk1_base import HARDWARE_FILE

FLAGS = flags.FLAGS
flags.DEFINE_bool("check_only", False, "No hardware: replay the demos through the limits in dk1_hardware.json.")
flags.DEFINE_string("demos", "data/stack_green_blue/success/*/traj.hdf5", "Demo traj.hdf5 glob ('' = none).")
flags.DEFINE_float("min_arm_distance", 0.08, "Closest allowed wrist/flange/tool0 pair of the two arms (m).")
flags.DEFINE_float("floor_margin", 0.005, "Floor = highest table touch (tool0 z) + this (m).")
flags.DEFINE_float("box_margin", 0.02, "Added to the swept range in x/y and on top (m).")
flags.DEFINE_integer("table_touches", 3, "Table touches per arm.")
flags.DEFINE_integer("tip_touches", 4, "Fingertip-to-fingertip touches (>= 3, not on one line).")
flags.DEFINE_string("out", "", "Write here instead of configs/task/dk1_hardware.json (tests).")
flags.DEFINE_float("max_fit_residual", 0.015, "Warn above this RMS residual (m) of the arm-to-arm fit.")


class Teleop:
    """Followers mirror the leaders in a background thread; records follower joints."""

    def __init__(self, env, controller):
        self.env, self.controller = env, controller
        self.state = env.get_state()
        self.log = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        dt = 1.0 / self.env.control_hz
        while not self._stop.is_set():
            t0 = time.time()
            target, _ = self.controller.forward()
            self.env.step(target)
            q = self.env.get_state()
            with self._lock:
                self.state = q
                if self.log is not None:
                    self.log.append(q)
            time.sleep(max(0.0, dt - (time.time() - t0)))

    def now(self) -> np.ndarray:
        with self._lock:
            return self.state.copy()

    def start_log(self):
        with self._lock:
            self.log = []

    def stop_log(self) -> np.ndarray:
        with self._lock:
            out, self.log = np.asarray(self.log), None
        return out

    def stop(self):
        self._stop.set()
        self.thread.join(timeout=2.0)


def _enter(msg: str):
    print(msg, flush=True)
    try:
        with open("/dev/tty") as tty:
            tty.readline()
    except OSError:
        input()


def _fmt_box(box) -> str:
    return "  ".join(f"{a} {lo:+.3f}..{hi:+.3f}" for a, (lo, hi) in zip("xyz", np.asarray(box)))


def report(safety: DK1Safety, demos: str):
    if not demos:
        return
    r = check_demos(safety, demos)
    print(f"\n[check] {r['episodes']} demos, {r['steps']} steps: safety held the arms on {r['held']} steps "
          f"({r['counts']})")
    for s in SIDES:
        print(f"[check] demo tool0 range {s:5s}: {_fmt_box(r['tool0_range'][s])}")
    if r["min_arm_distance"] is not None:
        print(f"[check] closest arm-to-arm distance in the demos: {r['min_arm_distance']:.3f} m "
              f"(limit {safety.min_dist})")
    if r["held"]:
        print("[check] WARNING: the limits would hold the arms during recorded demos. Look at the counts above.")
    return r


def record(env, controller) -> dict:
    fk = DK1Points()
    tele = Teleop(env, controller)
    try:
        # 1. table
        floor = {}
        for s in SIDES:
            zs = []
            for i in range(FLAGS.table_touches):
                _enter(f"\n[1/3 table] Touch the table with the {s.upper()} fingertips (spot {i + 1}/"
                       f"{FLAGS.table_touches}, spread out), hold still, press Enter.")
                z = fk(tele.now()[ARM_JOINTS[s]])[-1, 2]
                zs.append(z)
                print(f"  {s} tool0 z = {z:+.4f} m")
            floor[s] = max(zs) + FLAGS.floor_margin
            print(f"  {s} floor = {floor[s]:+.4f} m (touch spread {max(zs) - min(zs):.4f} m)")

        # 2. arm-to-arm transform
        pl, pr = [], []
        for i in range(FLAGS.tip_touches):
            _enter(f"\n[2/3 arms] Touch the LEFT and RIGHT fingertips together (spot {i + 1}/{FLAGS.tip_touches}: "
                   f"spread out, vary the height), hold still, press Enter.")
            q = tele.now()
            pl.append(fk(q[ARM_JOINTS['left']])[-1])
            pr.append(fk(q[ARM_JOINTS['right']])[-1])
        T, res = fit_rigid(np.array(pr), np.array(pl))
        rms = float(np.sqrt(np.mean(res ** 2)))
        print(f"  right base in left frame: t = {np.round(T[:3, 3], 3)}, residual per touch {np.round(res, 3)} "
              f"(RMS {rms:.4f} m)")
        spread = np.linalg.svd(np.array(pl) - np.mean(pl, 0), compute_uv=False)
        if rms > FLAGS.max_fit_residual or spread[1] < 0.05:
            print(f"  WARNING: poor fit (RMS {rms:.3f} m, 2nd spread {spread[1]:.3f} m). Rerun with touches "
                  f"further apart and the tips exactly together.")

        # 3. workspace sweep
        _enter("\n[3/3 workspace] Press Enter, then sweep BOTH arms over everything they may reach in the task "
               "(edges of the work area, highest lifts). Keep a hand clear.")
        tele.start_log()
        _enter("  Recording... press Enter when done.")
        sweep = tele.stop_log()
        print(f"  {len(sweep)} samples")
    finally:
        tele.stop()

    boxes = {}
    for s in SIDES:
        ee = np.array([fk(q[ARM_JOINTS[s]])[-1] for q in sweep]) if len(sweep) else np.zeros((0, 3))
        lo, hi = ee.min(0) - FLAGS.box_margin, ee.max(0) + FLAGS.box_margin
        boxes[s] = np.stack([lo, hi], 1)
        boxes[s][2, 0] = floor[s]
    sweep_min = DK1Safety(None, T, FLAGS.min_arm_distance)
    d = min(sweep_min.arm_distance(sweep_min.points(q)) for q in sweep) if len(sweep) else np.nan
    print(f"  closest arm-to-arm distance during the sweep: {d:.3f} m")
    return {"boxes": boxes, "T": T, "fit_rms": rms}


def main(_):
    cfg = FLAGS.task_config
    if FLAGS.check_only:
        safety = DK1Safety(cfg.get("ee_bounds"), cfg.get("right_base_in_left"), cfg.get("min_arm_distance"))
        if not safety.active:
            print(f"No Cartesian safety in {HARDWARE_FILE}; run without --check_only on the robot first.")
        report(safety, FLAGS.demos)
        return

    env_kwargs = dict(cfg)
    env_kwargs.update(video_dir="", env_usage="collect", ee_bounds=None, right_base_in_left=None)
    env = cfg.env(**env_kwargs)
    controller = cdk.LeaderMirror(env, cfg)
    try:
        env.reset()
        controller.reset_state()
        if not controller.mock:
            cdk._sync_follower_to_leaders(env, controller, FLAGS.sync_max_delta, FLAGS.sync_tolerance)
        rec = record(env, controller)
    finally:
        env.close()

    boxes = {s: b.copy() for s, b in rec["boxes"].items()}
    if FLAGS.demos:  # recorded demo motion must stay allowed: extend x/y/top (never the floor)
        r = check_demos(DK1Safety(), FLAGS.demos)
        for s in SIDES:
            rng = r["tool0_range"][s]
            boxes[s][:, 0] = np.minimum(boxes[s][:, 0], rng[:, 0])
            boxes[s][:, 1] = np.maximum(boxes[s][:, 1], rng[:, 1])
            boxes[s][2, 0] = rec["boxes"][s][2, 0]
    home = np.asarray(cfg.reset_joints, float)
    fk = DK1Points()
    for s in SIDES:
        h = fk(home[ARM_JOINTS[s]])[-1]
        inside = bool(np.all(h >= boxes[s][:, 0]) and np.all(h <= boxes[s][:, 1]))
        print(f"\n{s:5s} box: {_fmt_box(boxes[s])}   home tool0 {np.round(h, 3)} {'inside' if inside else 'OUTSIDE'}")

    bad = [s for s in SIDES if np.any(boxes[s][:, 0] >= boxes[s][:, 1])]
    if bad:
        raise SystemExit(f"Empty box for {bad} (floor above the top?): check the table touches and rerun. Nothing saved.")
    safety = DK1Safety({s: boxes[s] for s in SIDES}, rec["T"], FLAGS.min_arm_distance)
    report(safety, FLAGS.demos)
    home_d = safety.arm_distance(safety.points(home))
    print(f"arm-to-arm distance at home: {home_d:.3f} m")

    out = Path(FLAGS.out) if FLAGS.out else HARDWARE_FILE
    hw = json.loads(HARDWARE_FILE.read_text()) if HARDWARE_FILE.exists() else {}
    hw["ee_bounds"] = {s: np.round(boxes[s], 4).tolist() for s in SIDES}
    hw["right_base_in_left"] = np.round(rec["T"], 5).tolist()
    hw["min_arm_distance"] = FLAGS.min_arm_distance
    hw["safety_recorded"] = {"date": time.strftime("%Y-%m-%d %H:%M"), "fit_rms_m": round(rec["fit_rms"], 4)}
    if out.exists():
        shutil.copy(out, out.with_suffix(".json.bak"))
    out.write_text(json.dumps(hw, indent=2) + "\n")
    print(f"\nSaved to {out} (previous: {out.with_suffix('.json.bak').name}). "
          f"Commit it once the check above shows 0 held steps.")


if __name__ == "__main__":
    app.run(main)
