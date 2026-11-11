"""Record the DK1 Cartesian safety box (per arm, on tool0) and save it to dk1_hardware.json.

    source client_dk1/.venv/bin/activate
    python -m client.dk1_record_safety              # guided recording, ~2 min, teleop with the leaders
    python -m client.dk1_record_safety --refit      # no hardware: rebuild the box from the saved raw
                                                    # recording + demos (e.g. after changing a margin)
    python -m client.dk1_record_safety --check_only # no hardware: replay the demos through the saved box

Teleop as in demo collection (followers mirror the leader joints, safety off). Two steps:

  1. Table: touch the table with each gripper's CLOSED fingertips at 3 spread-out spots.
  2. Workspace: sweep both arms over everything they may reach in the task (`sweep_s` seconds).

No Enter needed while your hands are on the leaders (Enter still works everywhere):
  --capture grip (default): a touch is captured once the arm's fingers are closed (leader trigger
      squeezed; tool0 is between the fingertips) and it holds still for `hold_s`. Open the fingers
      or move on to re-arm for the next spot.
  --capture still: same without the closed-fingers condition.
  Step 2 starts on Enter or a squeeze of BOTH triggers, and stops after `sweep_s` or on another
  squeeze of BOTH triggers. Each capture rings the terminal bell.

The box, per arm:
  x/y/top = sweep range + `box_margin`, widened to the demos' commanded tool0 range.
  floor   = lowest table touch - `press_in`. The arms are impedance controlled, so a command a bit
            below the table only presses on it; what we stop is driving hard into it. Demos whose
            command goes lower still lower the floor (demo min - `demo_margin`), but never below
            lowest touch - `max_press_in`; anything past that is reported, not allowed.
The demos are then replayed through the box (state = current, action14 = command); the held-step
count should be 0. The raw recording (touch heights, sweep range) is saved next to the box, so
--refit can change margins/demos without the robot. Previous file kept as .json.bak.
"""

from __future__ import annotations

import json
import select
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
from absl import app, flags

from client import collect_data_dk1 as cdk  # defines --task_config, --sync_max_delta, --sync_tolerance
from client.envs.dk1_safety import ARM_JOINTS, SIDES, DK1Points, DK1Safety, check_demos
from configs.task.dk1_base import HARDWARE_FILE

FLAGS = flags.FLAGS
flags.DEFINE_bool("check_only", False, "No hardware: replay the demos through the box in dk1_hardware.json.")
flags.DEFINE_bool("refit", False, "No hardware: rebuild the box from the raw recording saved in dk1_hardware.json.")
flags.DEFINE_string("demos", "data/dk1_pnp_bowl_new/success/*/traj.hdf5", "Demo traj.hdf5 glob ('' = none).")
flags.DEFINE_float("press_in", 0.01, "Floor = lowest table touch (tool0 z) - this (m): how far a command may "
                   "go below the table (impedance: it only presses).")
flags.DEFINE_float("max_press_in", 0.02, "Demos may lower the floor down to lowest touch - this (m), no further.")
flags.DEFINE_float("demo_margin", 0.003, "Floor/box kept this far (m) beyond the demos' commanded tool0.")
flags.DEFINE_float("box_margin", 0.02, "Added to the swept range in x/y and on top (m).")
flags.DEFINE_integer("table_touches", 3, "Table touches per arm.")
flags.DEFINE_string("out", "", "Write here instead of configs/task/dk1_hardware.json (tests).")
flags.DEFINE_enum("capture", "grip", ["grip", "still"], "Step 1: grip = fingers closed + still; still = "
                  "still only (Enter always captures).")
flags.DEFINE_float("trigger_s", 0.3, "Step 2: how long (s) the both-trigger start/stop squeeze must be held.")
flags.DEFINE_float("grip_closed", 0.5, "Leader trigger above this (0 open .. 1 closed) = squeezed.")
flags.DEFINE_float("grip_open", 0.3, "Leader trigger below this = released.")
flags.DEFINE_float("hold_s", 1.5, "Step 1: hold still this long (s) to capture a touch.")
flags.DEFINE_float("still_tol", 0.01, "Step 1: max joint range (rad) over hold_s that counts as still.")
flags.DEFINE_float("move_tol", 0.1, "Step 1: the arm must first move this far (rad, any joint) from the previous "
                   "capture, or open the fingers, so a capture never fires twice at the same spot.")
flags.DEFINE_float("sweep_s", 60.0, "Step 2: workspace sweep duration (s); Enter or both triggers end it early.")


class Teleop:
    """Followers mirror the leaders in a background thread; records follower joints."""

    def __init__(self, env, controller):
        self.env, self.controller = env, controller
        self.state = env.get_state()
        self.grips = np.zeros(2)  # leader triggers [L, R], 0 open .. 1 closed
        self.log = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        dt = 1.0 / self.env.control_hz
        while not self._stop.is_set():
            t0 = time.time()
            target, leader = self.controller.forward()
            self.env.step(target)
            q = self.env.get_state()
            with self._lock:
                self.state = q
                self.grips = np.asarray(leader, float)[[6, 13]]
                if self.log is not None:
                    self.log.append(q)
            time.sleep(max(0.0, dt - (time.time() - t0)))

    def now(self) -> np.ndarray:
        with self._lock:
            return self.state.copy()

    def triggers(self) -> np.ndarray:
        with self._lock:
            return self.grips.copy()

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


class _Keys:
    """Non-blocking "was Enter pressed?" on the controlling terminal (stdin as a fallback)."""

    def __enter__(self):
        try:
            self.f = open("/dev/tty")
        except OSError:
            self.f = sys.stdin
        return self

    def pressed(self) -> bool:
        try:
            if select.select([self.f], [], [], 0)[0]:
                self.f.readline()
                return True
        except (OSError, ValueError):
            pass
        return False

    def __exit__(self, *exc):
        if self.f is not sys.stdin:
            self.f.close()


def _beep(msg: str):
    print(f"\a  >>> {msg}", flush=True)


class _Squeeze:
    """Squeeze detector: fires once a trigger set is held > grip_closed for trigger_s, after all
    triggers were first released (< grip_open). `onset` = when that squeeze started."""

    def __init__(self, need_both: bool):
        self.need_both = need_both
        self.armed = False
        self.onset = None

    def update(self, g: np.ndarray, now: float) -> bool:
        if np.all(g < FLAGS.grip_open):
            self.armed, self.onset = True, None
            return False
        closed = np.all(g > FLAGS.grip_closed) if self.need_both else np.any(g > FLAGS.grip_closed)
        if not (self.armed and closed):
            self.onset = None
            return False
        if self.onset is None:
            self.onset = now
        if now - self.onset >= FLAGS.trigger_s:
            self.armed = False
            return True
        return False


def _wait_capture(tele: "Teleop", sides, msg: str, prev: np.ndarray | None) -> np.ndarray:
    """Capture once the `sides` arms are re-armed (moved >= move_tol from `prev`, or fingers opened)
    and then held still for hold_s (range <= still_tol per joint) -- with --capture grip, with those
    arms' leader triggers squeezed (fingers closed) the whole time. Enter captures immediately.
    Returns the 14-D state (mean over the still window)."""
    need_closed = FLAGS.capture == "grip"
    idx = np.concatenate([np.arange(14)[ARM_JOINTS[s]] for s in sides])
    gi = [SIDES.index(s) for s in sides]
    how = "close the fingers (squeeze the trigger" + ("s" if len(sides) > 1 else "") + "), touch, hold still" \
        if need_closed else "touch, hold still"
    print(msg + f"  [{how} {FLAGS.hold_s:.1f} s = capture; Enter = capture now]", flush=True)
    dt = 0.02
    n_hold = max(2, int(round(FLAGS.hold_s / dt)))
    window = []
    armed = prev is None
    last_note = 0.0
    with _Keys() as keys:
        while True:
            q, g = tele.now(), tele.triggers()[gi]
            if keys.pressed():
                _beep("captured (Enter)")
                return q
            if not armed and (np.max(np.abs(q[idx] - prev[idx])) >= FLAGS.move_tol or np.any(g < FLAGS.grip_open)):
                armed = True
            closed = bool(np.all(g > FLAGS.grip_closed)) or not need_closed
            window = (window + [q])[-n_hold:] if closed else []
            if armed and len(window) == n_hold:
                w = np.asarray(window)[:, idx]
                rng = float(np.max(w.max(0) - w.min(0)))
                if rng <= FLAGS.still_tol:
                    _beep("captured, move on to the next spot")
                    return np.asarray(window).mean(0)
                if time.time() - last_note > 5.0 and rng < 5 * FLAGS.still_tol:
                    print(f"  (almost still: joint range {rng:.3f} rad > still_tol {FLAGS.still_tol})", flush=True)
                    last_note = time.time()
            time.sleep(dt)


def _fmt_box(box) -> str:
    return "  ".join(f"{a} {lo:+.3f}..{hi:+.3f}" for a, (lo, hi) in zip("xyz", np.asarray(box)))


def report(safety: DK1Safety, demos: str):
    if not demos:
        return
    r = check_demos(safety, demos)
    print(f"\n[check] {r['episodes']} demos, {r['steps']} steps: safety held the arms on {r['held']} steps "
          f"({r['counts']})")
    for s in SIDES:
        print(f"[check] demo commanded tool0 range {s:5s}: {_fmt_box(r['tool0_range'][s])}")
    if r["held"]:
        print("[check] WARNING: the limits would hold the arms during recorded demos. Look at the counts above.")
    return r


def record(env, controller) -> dict:
    """Teleop recording. Returns the raw recording: per arm the table-touch tool0 z values and the
    swept tool0 range ([[x], [y], [z]] x [lo, hi])."""
    fk = DK1Points()
    tele = Teleop(env, controller)
    try:
        table_z = {}
        prev = tele.now()
        for s in SIDES:
            zs = []
            for i in range(FLAGS.table_touches):
                prev = _wait_capture(tele, [s], f"\n[1/2 table] Touch the table with the {s.upper()} fingertips "
                                     f"(spot {i + 1}/{FLAGS.table_touches}, spread out).", prev)
                zs.append(float(fk(prev[ARM_JOINTS[s]])[-1, 2]))
                print(f"  {s} tool0 z = {zs[-1]:+.4f} m")
            table_z[s] = zs
            print(f"  {s} table: lowest {min(zs):+.4f} m, spread {max(zs) - min(zs):.4f} m")

        print(f"\n[2/2 workspace] Enter or squeeze BOTH triggers to start, then sweep BOTH arms over everything "
              f"they may reach in the task (edges of the work area, highest lifts) for {FLAGS.sweep_s:.0f} s. "
              f"Keep a hand clear.", flush=True)
        sq = _Squeeze(need_both=True)
        with _Keys() as keys:
            while not keys.pressed() and not sq.update(tele.triggers(), time.time()):
                time.sleep(0.02)
        tele.start_log()
        _beep(f"recording the sweep for {FLAGS.sweep_s:.0f} s (Enter or BOTH triggers = stop early)")
        t_end = time.time() + FLAGS.sweep_s
        next_note = 10.0 * np.floor(FLAGS.sweep_s / 10.0)
        sq = _Squeeze(need_both=True)  # armed only once the triggers are released after the start squeeze
        with _Keys() as keys:
            while (time.time() < t_end and not keys.pressed()
                   and not sq.update(tele.triggers(), time.time())):
                left = t_end - time.time()
                if left <= next_note and next_note > 0:
                    print(f"  {next_note:.0f} s left", flush=True)
                    next_note -= 10.0
                time.sleep(0.05)
        sweep = tele.stop_log()
        _beep("sweep done")
        print(f"  {len(sweep)} samples")
    finally:
        tele.stop()
    if not len(sweep):
        raise SystemExit("Empty sweep: nothing saved.")
    sweep_range = {}
    for s in SIDES:
        ee = np.array([fk(q[ARM_JOINTS[s]])[-1] for q in sweep])
        sweep_range[s] = np.stack([ee.min(0), ee.max(0)], 1).round(4).tolist()
    return {"table_z": table_z, "sweep_range": sweep_range}


def legacy_raw(hw: dict) -> dict:
    """Raw recording for a dk1_hardware.json from before raw values were saved (floor was highest
    touch + 5 mm): the old box stands in for the sweep range, floor - 5 mm for the highest touch."""
    raw = {"table_z": {}, "sweep_range": {}}
    for s in SIDES:
        b = np.asarray(hw["ee_bounds"][s], float)
        raw["table_z"][s] = [float(b[2, 0] - 0.005)]
        b[:2, 0] += FLAGS.box_margin
        b[:2, 1] -= FLAGS.box_margin
        b[2, 1] -= FLAGS.box_margin
        raw["sweep_range"][s] = b.tolist()
    return raw


def build_boxes(raw: dict) -> dict:
    """Box per arm from the raw recording + demos (see the module docstring)."""
    demo = check_demos(DK1Safety(), FLAGS.demos)["tool0_range"] if FLAGS.demos else None
    boxes = {}
    for s in SIDES:
        sw = np.asarray(raw["sweep_range"][s], float)
        box = np.stack([sw[:, 0] - FLAGS.box_margin, sw[:, 1] + FLAGS.box_margin], 1)
        table = min(raw["table_z"][s])
        floor = table - FLAGS.press_in
        if demo is not None:
            d = demo[s]
            box[:, 0] = np.minimum(box[:, 0], d[:, 0] - FLAGS.demo_margin)
            box[:, 1] = np.maximum(box[:, 1], d[:, 1] + FLAGS.demo_margin)
            need = d[2, 0] - FLAGS.demo_margin
            if need < floor:
                floor = max(need, table - FLAGS.max_press_in)
                print(f"  {s}: demos command tool0 down to {d[2, 0]:+.4f} m, floor lowered to {floor:+.4f} m"
                      + ("" if floor <= need else f" (capped at lowest touch - max_press_in {FLAGS.max_press_in}: "
                         f"those demo steps stay held; check the table touches)"))
        box[2, 0] = floor
        print(f"  {s}: lowest table touch {table:+.4f} m -> floor {floor:+.4f} m ({(table - floor) * 1000:.0f} mm "
              f"below the table)")
        boxes[s] = box
    return boxes


def main(_):
    cfg = FLAGS.task_config
    if FLAGS.check_only:
        safety = DK1Safety(cfg.get("ee_bounds"))
        if not safety.active:
            print(f"No Cartesian safety in {HARDWARE_FILE}; run without --check_only on the robot first.")
        report(safety, FLAGS.demos)
        return

    hw = json.loads(HARDWARE_FILE.read_text()) if HARDWARE_FILE.exists() else {}
    if FLAGS.refit:
        raw = hw.get("safety_recorded", {}).get("raw")
        if raw is None:
            if "ee_bounds" not in hw:
                raise SystemExit(f"Nothing recorded in {HARDWARE_FILE}; run on the robot first.")
            print("No raw recording saved (older file): using the old box as the sweep range and its floor - 5 mm "
                  "as the highest table touch. Re-record for exact values.")
            raw = legacy_raw(hw)
    else:
        env_kwargs = dict(cfg)
        env_kwargs.update(video_dir="", env_usage="collect", ee_bounds=None)
        env = cfg.env(**env_kwargs)
        controller = cdk.LeaderMirror(env, cfg)
        try:
            env.reset()
            controller.reset_state()
            if not controller.mock:
                cdk._sync_follower_to_leaders(env, controller, FLAGS.sync_max_delta, FLAGS.sync_tolerance)
            raw = record(env, controller)
        finally:
            env.close()

    print()
    boxes = build_boxes(raw)
    home = np.asarray(cfg.reset_joints, float)
    fk = DK1Points()
    for s in SIDES:
        h = fk(home[ARM_JOINTS[s]])[-1]
        inside = bool(np.all(h >= boxes[s][:, 0]) and np.all(h <= boxes[s][:, 1]))
        print(f"\n{s:5s} box: {_fmt_box(boxes[s])}   home tool0 {np.round(h, 3)} {'inside' if inside else 'OUTSIDE'}")
    bad = [s for s in SIDES if np.any(boxes[s][:, 0] >= boxes[s][:, 1])]
    if bad:
        raise SystemExit(f"Empty box for {bad} (floor above the top?): check the table touches and rerun. Nothing saved.")
    report(DK1Safety({s: boxes[s] for s in SIDES}), FLAGS.demos)

    out = Path(FLAGS.out) if FLAGS.out else HARDWARE_FILE
    for k in ("right_base_in_left", "min_arm_distance"):  # arm-to-arm check removed
        hw.pop(k, None)
    hw["ee_bounds"] = {s: np.round(boxes[s], 4).tolist() for s in SIDES}
    rec = hw.get("safety_recorded", {}) if FLAGS.refit else {"date": time.strftime("%Y-%m-%d %H:%M")}
    rec.pop("fit_rms_m", None)
    rec.update(raw=raw, params={k: getattr(FLAGS, k) for k in
                                ("press_in", "max_press_in", "demo_margin", "box_margin", "demos")})
    if FLAGS.refit:
        rec["refit"] = time.strftime("%Y-%m-%d %H:%M")
    hw["safety_recorded"] = rec
    if out.exists():
        shutil.copy(out, out.with_suffix(".json.bak"))
    out.write_text(json.dumps(hw, indent=2) + "\n")
    print(f"\nSaved to {out} (previous: {out.with_suffix('.json.bak').name}). "
          f"Commit it once the check above shows 0 held steps.")


if __name__ == "__main__":
    app.run(main)
