"""Real-robot intervention test: put the follower in a demo pose, then take over with the GELLO leaders.

Uses the exact RL-rollout path: DK1Env.get_human_action() (foot pedal + anchored relative IK thread)
-> DK1Env.step() (joint limits, per-step joint cap, EE box). Nothing else commands the arms.

  1. Arms go home (folded), then retrace the recorded follower path of demo D up to frame T,
     slowly (--path_speed rad/s). The arm only visits poses it already visited in this scene.
  2. The follower HOLDS that pose. HOLD the foot pedal = take over (anchors; the follower must
     not move), move the GELLO, release = hand back (follower holds). Repeat as often as you like.
  3. Ctrl+C: release, retrace the demo path back to frame 0, open grippers, home.

Live print (2 Hz while engaged), per arm: |leader EE delta - follower EE delta| since engage
(should stay < ~1-2 cm while moving slowly), joint-cap hits, IK thread ms. A log .npz is saved.

    python -m client.teleop.test_intervention_hw --list                         # pick D/T
    python -m client.teleop.test_intervention_hw --demo 11 --frame 120          # real robot
    python -m client.teleop.test_intervention_hw --mock --demo 11 --frame 120   # no hardware
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import h5py
import numpy as np

SIDES = {"left": slice(0, 7), "right": slice(7, 14)}
ARM = np.r_[0:6, 7:13]


def load_demo(data, d):
    with h5py.File(os.path.join(data, str(d), "traj.hdf5")) as f:
        return f["saved_observation/state"][:].astype(np.float64), f["action/leader14"][:].astype(np.float64)


def list_demos(data):
    for p in sorted(glob.glob(os.path.join(data, "*", "traj.hdf5")), key=lambda s: int(s.split(os.sep)[-2])):
        d = p.split(os.sep)[-2]
        s, _ = load_demo(data, d)
        closed = {side: np.flatnonzero(s[:, sl][:, 6] > 0.3) for side, sl in SIDES.items()}
        grasp = ", ".join(f"{k} grip>0.3 frames {v[0]}-{v[-1]}" for k, v in closed.items() if len(v)) or "no grasp"
        print(f"demo {d:>3}: {len(s)} frames | {grasp}")


def densify(path, speed, hz):
    """Joint-space waypoints -> commands at `hz` with max joint speed `speed` (rad/s)."""
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.abs(b - a)[ARM].max() / (speed / hz))))
        out += [a + (b - a) * (i / n) for i in range(1, n + 1)]
    return np.array(out)


def play(env, cmds, hz):
    for c in cmds:
        t0 = time.perf_counter()
        env._arms.send(c)
        env._last_command = c.copy()
        if not env.mock:
            time.sleep(max(0.0, 1.0 / hz - (time.perf_counter() - t0)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task_config", default="configs/task/dk1_task.py")
    p.add_argument("--data", default="data/stack_green_blue/success")
    p.add_argument("--list", action="store_true")
    p.add_argument("--demo", type=int, default=11)
    p.add_argument("--frame", type=int, default=120)
    p.add_argument("--path_speed", type=float, default=0.3, help="rad/s when retracing the demo path")
    p.add_argument("--max_delta", type=float, default=0.05,
                   help="per-step joint cap (rad/step at control_hz) for human actions in this test")
    p.add_argument("--mock", action="store_true", help="fake arms/leaders/cams, scripted pedal + leader motion")
    p.add_argument("--log", default="")
    args = p.parse_args()

    if args.list:
        list_demos(args.data)
        return 0

    from client.run_client import load_task_config
    from client.envs.dk1_env import clip_to_limits

    cfg = load_task_config("configs/task/dk1_mock.py" if args.mock else args.task_config)
    hz = float(cfg.control_hz)
    from client.teleop.mink_ik import DK1Kinematics
    kin = DK1Kinematics()
    state, leader_rec = load_demo(args.data, args.demo)
    T = int(np.clip(args.frame, 0, len(state) - 1))
    path = np.array([clip_to_limits(s) for s in state[: T + 1]])
    target = path[-1]
    home = np.asarray(cfg.reset_joints, float).reshape(14).copy()
    home[6] = home[13] = 0.0
    if np.abs(path[0] - home)[ARM].max() > 0.15:
        print(f"!! demo {args.demo} frame 0 is {np.abs(path[0]-home)[ARM].max():.2f} rad from home; "
              f"the first leg (home -> frame 0) is a straight joint-space move.")

    env_kwargs = dict(cfg)
    env_kwargs.update(env_usage="eval", wait_for_scene_reset=False, max_joint_delta_per_step=args.max_delta)
    env = cfg.env(**env_kwargs)  # real: connects followers, cameras, leaders, foot pedal
    log = {k: [] for k in ("t", "engaged", "leader", "follower", "command", "executed")}
    cap_hits = cap_steps = 0
    try:
        print(f"\nDemo {args.demo}, frame {T}/{len(state)-1}. Target follower joints:\n"
              f"  L {np.round(target[0:7], 2)}\n  R {np.round(target[7:14], 2)}")
        print(f"Plan: home -> retrace demo path ({len(path)} frames) at <= {args.path_speed} rad/s -> HOLD.")
        if not args.mock:
            input("Hand on the e-stop. Press Enter to move home.")
        env.reset()
        if not args.mock:
            input("Scene set up as at that demo frame (e.g. block in the gripper if it is mid-grasp)? "
                  "Press Enter to retrace the demo path.")
        play(env, densify(np.vstack([env.get_state(), path]), args.path_speed, 50.0), 50.0)
        time.sleep(0.5)
        err = np.abs(env.get_state() - target)[ARM].max()
        print(f"At demo pose (max joint error {err:.3f} rad). Follower holds.")
        print("Now: HOLD the pedal to take over, move the GELLO slowly, release to hand back. "
              "Ctrl+C here to finish." if not args.mock else "Mock: scripted engage / motion / release.")

        thread = env._ensure_intervention()
        ctrl = thread.controller
        anchor = None
        last_print, step, t_start = 0.0, 0, time.perf_counter()
        mock_lead0 = None
        idle_ref = None  # leader reading when the test loop started (released-state motion check)
        while True:
            t0 = time.perf_counter()
            tt = t0 - t_start
            if args.mock:  # scripted operator: engage at 1 s, wiggle the leaders, release at 5 s, stop at 6 s
                lead = env.get_leaders()
                if mock_lead0 is None and tt > 1.0:
                    mock_lead0 = env.get_state() + np.r_[[0.3] * 6, 0.0, [-0.2] * 6, 0.0]  # leader != follower
                    lead.pose = mock_lead0.copy()
                    env.get_pedal().set_engaged(True)
                if mock_lead0 is not None and env.get_pedal().engaged:
                    w = 0.2 * np.sin(2 * np.pi * 0.3 * (tt - 1.0))
                    lead.pose = mock_lead0 + np.r_[[w, w / 2, -w / 2, 0, w, 0], 0, [-w] * 6, 0]
                if tt > 5.0:
                    env.get_pedal().set_engaged(False)
                if tt > 6.0:
                    break
            action, is_human = env.get_human_action()
            # Engaged: reuse the IK thread's reading (no extra bus traffic). Released: read directly
            # (bus is locked), so you can check the GELLO reads change BEFORE taking over.
            if is_human and thread.last_leader is not None:
                leader_now = thread.last_leader.copy()
            else:
                try:
                    leader_now = env.get_leaders().get()
                except Exception as e:  # never abort the test on a transient leader read error
                    print(f"[test] leader read error {e!r}; reusing last reading", flush=True)
                    leader_now = log["leader"][-1] if log["leader"] else None
            if leader_now is None:
                continue
            if idle_ref is None:
                idle_ref = leader_now.copy()
            if is_human and anchor is None:
                anchor = {"lead0": leader_now.copy(), "fol0": env.get_state().copy()}
                print("[test] engaged: the follower should NOT have moved.", flush=True)
            if not is_human and anchor is not None:
                print("[test] released: follower holds.", flush=True)
                anchor = None
            cmd = action if is_human else -np.ones(14)
            before = env._clip_count
            executed = env.step(cmd)["executed_action"]
            if is_human:
                cap_steps += 1
                cap_hits += env._clip_count - before
            fol = env.get_state()
            for k, v in zip(log, (tt, is_human, leader_now, fol, cmd, executed)):
                log[k].append(v)
            if is_human and anchor is not None and t0 - last_print > 0.5:
                last_print = t0
                parts = []
                lf0 = ctrl.leader_to_follower_joints(anchor["lead0"])
                lfn = ctrl.leader_to_follower_joints(leader_now)
                for side, sl in SIDES.items():
                    fk = kin.fk
                    dl = fk(lfn[sl][:6])[0] - fk(lf0[sl][:6])[0]
                    df = fk(fol[sl][:6])[0] - fk(anchor["fol0"][sl][:6])[0]
                    parts.append(f"{side[0].upper()}: leader moved {np.linalg.norm(dl)*100:4.1f} cm, "
                                 f"follower err {np.linalg.norm(dl-df)*100:4.1f} cm, grip {fol[sl][6]:.2f}")
                print(" | ".join(parts) + f" | cap {100*cap_hits/max(cap_steps,1):.0f}% | ik {thread.loop_ms:.1f} ms",
                      flush=True)
            if not is_human and t0 - last_print > 1.0:
                last_print = t0
                lr0, lrn = ctrl.leader_to_follower_joints(idle_ref), ctrl.leader_to_follower_joints(leader_now)
                moved = [np.linalg.norm(kin.fk(lrn[sl][:6])[0] - kin.fk(lr0[sl][:6])[0])
                         for sd, sl in SIDES.items()]
                print(f"[released] GELLO moved since start: L {moved[0]*100:4.1f} cm, R {moved[1]*100:4.1f} cm, "
                      f"triggers {leader_now[6]:.2f}/{leader_now[13]:.2f}  (wiggle it: these must change)", flush=True)
            step += 1
            time.sleep(max(0.0, 1.0 / hz - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        try:
            env._release_intervention()
            print(f"Joint cap hit on {cap_hits}/{cap_steps} human steps.")
            if not args.mock:
                input("Hand back to retracing: the arm returns along the demo path to its start, then home. "
                      "Clear the workspace / hold the block if needed, then press Enter.")
            back = np.vstack([env.get_state(), path[::-1]])
            back[1:, 6] = back[1:, 13] = 0.0  # open grippers on the way back
            play(env, densify(back, args.path_speed, 50.0), 50.0)
            env.home()
        finally:
            out = args.log or f"outputs/intervention_test_demo{args.demo}_f{T}_{time.strftime('%Y%m%d_%H%M%S')}.npz"
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez(out, **{k: np.asarray(v) for k, v in log.items()})
            print(f"log: {out}")
            env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
