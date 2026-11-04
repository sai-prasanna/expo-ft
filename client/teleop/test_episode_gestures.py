"""Checks for the hands-free episode control (no hardware).

    python -m client.teleop.test_episode_gestures     # exits 1 on failure

1. EpisodeGestureFSM on scripted leader paths (keep / discard / idle / start / loose home).
2. collect_data_dk1 end to end on the mock DK1 env, with gesture control forced on and a
   scripted leader: two episodes back to back without homing, the first kept, the second
   discarded. Checks that the kept traj.hdf5 starts and ends at home (reset_joints).
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

from client.teleop.episode_gestures import EpisodeGestureFSM

HZ = 15.0
ARM = np.r_[0:6, 7:13]
OPEN, CLOSED = np.zeros(2), np.ones(2)


def hold(q, grips, seconds):
    return [(np.asarray(q, float), np.asarray(grips, float))] * int(round(seconds * HZ))


def move(a, b, seconds, grips=OPEN):
    n = int(round(seconds * HZ))
    return [(a + (b - a) * (i + 1) / n, np.asarray(grips, float)) for i in range(n)]


def task(rest, away=0.8, through_rest=False):
    """Leave the start pose, do something far away, come back."""
    far = rest + away
    path = move(rest, far, 1.5) + hold(far, [0, 1], 1.0)
    if through_rest:  # swing back through the start pose without stopping, then out again
        path += move(far, rest, 1.0) + move(rest, far, 1.0)
    return path + move(far, rest, 1.5)


def events(fsm, path):
    out = []
    for q, g in path:
        e = fsm.update(q, g)
        if e:
            out.append(e)
    return out


def fsm_cases() -> list[str]:
    rest = np.full(12, 0.1)
    fails = []

    def check(name, path, expect, home=None):
        got = events(EpisodeGestureFSM(HZ, home=home), path)
        ok = got == expect
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {got}")
        if not ok:
            fails.append(name)

    idle = hold(rest, OPEN, 1.0)  # at home, still, triggers open -> ready
    stop = hold(rest, OPEN, 0.7)
    keep = hold(rest, CLOSED, 0.8)
    E = ["ready", "start", "stop"]
    check("keep", idle + task(rest) + stop + keep, E + ["keep"])
    check("discard (left trigger alone)", idle + task(rest) + stop + hold(rest, [1, 0], 0.8), E + ["discard"])
    check("right trigger alone does nothing", idle + task(rest) + stop + hold(rest, [0, 1], 1.5), E)
    check("half squeeze (0.6) counts", idle + task(rest) + stop + hold(rest, [0.6, 0.6], 0.8), E + ["keep"])
    check("too-short squeeze does nothing", idle + task(rest) + stop + hold(rest, CLOSED, 0.3) + hold(rest, OPEN, 1), E)
    check("one then both -> keep", idle + task(rest) + stop + hold(rest, [1, 0], 0.3) + keep, E + ["keep"])
    check("waits for the gesture however long", idle + task(rest) + stop + hold(rest, OPEN, 30.0), E)
    check("moving during the decision does nothing", idle + task(rest) + stop + task(rest) + stop + keep, E + ["keep"])
    check("passing through home mid-task does not stop", idle + task(rest, through_rest=True) + stop, E)

    # After the decision: idle until an arm clearly moves, however long the scene reset takes.
    drift = move(rest, rest + 0.3, 60.0)  # 0.005 rad/s sag over a minute
    check("after keep: idle for 60 s (triggers released, slow drift) never starts",
          idle + task(rest) + stop + keep + hold(rest, OPEN, 60.0) + drift, E + ["keep", "ready"])
    check("after keep: moving an arm starts the next episode",
          idle + task(rest) + stop + keep + hold(rest, OPEN, 5.0) + task(rest) + stop + keep,
          E + ["keep", "ready", "start", "stop", "keep"])
    check("small nudge (0.05 rad) does not start", idle + move(rest, rest + 0.05, 0.3) + hold(rest + 0.05, OPEN, 3.0),
          ["ready"])
    check("triggers never start an episode", idle + hold(rest, CLOSED, 1.0) + hold(rest, OPEN, 1.0), ["ready"])
    check("slow start near home (the hardware log), then the real task",
          idle + move(rest, rest + 0.28, 0.4) + move(rest + 0.28, rest + 0.22, 0.2) + hold(rest + 0.22, OPEN, 1.0)
          + task(rest + 0.22) + move(rest + 0.22, rest, 0.5) + stop + keep, E + ["keep"])

    # Loose home: wrists ignored, shoulder/elbow within 0.4.
    sloppy = rest.copy()
    sloppy[[3, 4, 5, 9, 10, 11]] += 1.0  # wrists anywhere
    sloppy[[1, 7]] += 0.35  # shoulders 0.35 off
    check("home with wrists anywhere and shoulders 0.35 off still stops",
          idle + task(rest) + move(rest + 0.8, sloppy, 0.5) + hold(sloppy, OPEN, 0.7) + hold(sloppy, CLOSED, 0.8),
          E + ["keep"])

    # Fixed home (as in the collector).
    home = np.zeros(12)
    lifted = home.copy()
    lifted[7] = 1.10  # right shoulder up
    check("fixed home: not ready while an arm is lifted", hold(lifted, OPEN, 4.0), [], home=home)
    check("fixed home: ready once back home, episode ends at home",
          hold(lifted, OPEN, 1.0) + move(lifted, home, 1.0) + hold(home, OPEN, 1.0)
          + task(home) + hold(home, OPEN, 0.7) + hold(home, CLOSED, 0.8), E + ["keep"], home=home)
    return fails


def collector_end_to_end() -> list[str]:
    from absl import flags

    import client.collect_data_dk1 as cd
    from configs.task import dk1_mock

    tmp = tempfile.mkdtemp(prefix="gesture_test_")
    flags.FLAGS([sys.argv[0], f"--save_root={tmp}", "--task_config=configs/task/dk1_mock.py"])
    cfg = dk1_mock.get_config()
    cfg.wait_for_scene_reset = False
    flags.FLAGS.task_config = cfg
    env_kwargs = dict(cfg, video_dir="", env_usage="collect")
    env = cfg.env(**env_kwargs)
    env.ignore_auto_reset = True
    controller = cd.LeaderMirror(env, cfg)
    controller.mock = False  # take the real (gesture) path; leaders/arms are still the mock ones
    home = env.get_state().copy()
    rest = home[ARM]

    def to14(q12, g):
        out = home.copy()
        out[ARM], out[[6, 13]] = q12, g
        return out

    first = hold(rest, OPEN, 2.0) + task(rest, away=0.8) + hold(rest, OPEN, 1.0) + hold(rest, CLOSED, 1.0)
    second = hold(rest, OPEN, 3.0) + task(rest, away=0.8) + hold(rest, OPEN, 1.0) + hold(rest, [1, 0], 1.0)
    script = iter([to14(q, g) for q, g in first + second])

    def get_leader():
        try:
            controller.leaders.pose = next(script)
        except StopIteration:
            raise RuntimeError("script ran out before the episode ended")
        return controller.leaders.get()

    controller.get_leader = get_leader
    fails = []
    try:
        r1 = cd.run_and_route_one(env, controller, tmp)
        r2 = cd.run_and_route_one(env, controller, tmp)
    finally:
        env.close()
    ok = r1["dest_dir"] is not None and r1["result"].get("end_reason") == "keep"
    print(f"[{'ok' if ok else 'FAIL'}] collector episode 1 kept: {r1['result']}")
    fails += [] if ok else ["collector keep"]
    ok = r2["dest_dir"] is None and r2["result"].get("end_reason") == "discard"
    print(f"[{'ok' if ok else 'FAIL'}] collector episode 2 discarded: {r2['result']}")
    fails += [] if ok else ["collector discard"]
    if r1["dest_dir"]:
        import h5py

        with h5py.File(os.path.join(r1["dest_dir"], "traj.hdf5")) as f:
            a = f["action/action14"][:]
            reason = f.attrs.get("end_reason")
        d0 = np.abs(a[0, ARM] - rest).max()
        d1 = np.abs(a[-1, ARM] - rest).max()
        peak = np.abs(a[:, ARM] - rest).max()
        n_rest = int(np.argmax(np.abs(a[:, ARM] - rest).max(1) > 1e-3))  # pre-roll frames at rest
        ok = d0 < 0.1 and d1 < 0.1 and peak > 0.4 and reason == "keep" and n_rest >= 5
        print(f"[{'ok' if ok else 'FAIL'}] kept traj: {len(a)} steps ({n_rest} at rest first = pre-roll), "
              f"start/end/peak distance from home {d0:.2f}/{d1:.2f}/{peak:.2f} rad, end_reason={reason}")
        fails += [] if ok else ["collector traj"]
    print(f"(test data under {tmp})")
    return fails


def main() -> int:
    fails = fsm_cases() + collector_end_to_end()
    print("\nALL OK" if not fails else f"\nFAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
