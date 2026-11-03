"""Standalone checks for the anchored relative-IK intervention.

Synthetic (no hardware): drives a fake leader along a smooth path, runs the 60 Hz IK and
reports the tracking error |FK(IK(target)) - target|. Optionally shows it in the MuJoCo viewer.

    python -m client.teleop.test_relative_ik                # synthetic, prints errors, exits 1 on failure
    python -m client.teleop.test_relative_ik --viewer       # + MuJoCo viewer (left arm)

Hardware dry run (NOTHING is sent to the followers):

    python -m client.teleop.test_relative_ik --hardware --task_config configs/task/dk1_task.py

  1. Mirror the leaders onto the followers with your usual teleop (or line them up by hand),
     then compare FK(leader) vs FK(follower) printed here: they should agree to ~1 cm / few
     degrees. If not, set intervention.leader_joint_offsets / leader_joint_signs.
  2. Press Enter to anchor, move the leaders, and watch the printed joint targets.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from client.teleop.relative_ik import RelativeIKController, quat_angle


def synthetic(args) -> int:
    home = np.array([0.0, 0.8, 1.2, 0.0, 0.0, 0.0, 0.0] * 2)
    ctrl = RelativeIKController(home14=home, dt=1.0 / args.rate)
    follower = home.copy()
    leader0 = home.copy()
    leader0[0:6] += np.array([0.3, 0.2, -0.2, 0.1, 0.3, -0.2])  # leader starts somewhere else
    ctrl.anchor(leader0, follower)

    viewer = None
    if args.viewer:
        import mujoco
        import mujoco.viewer
        model = ctrl.solvers["left"].model
        data = mujoco.MjData(model)
        viewer = mujoco.viewer.launch_passive(model, data)

    n = int(args.seconds * args.rate)
    pos_err, rot_err, final = [], [], None
    for i in range(n):
        t = i / args.rate
        leader = leader0.copy()
        # Smooth joint-space wiggle on the leader -> SE(3) deltas of several cm / tens of degrees.
        leader[0:6] += 0.25 * np.sin(2 * np.pi * 0.25 * t) * np.array([1.0, 0.5, -0.6, 0.8, 0.7, 1.0])
        leader[7:13] += 0.20 * np.sin(2 * np.pi * 0.2 * t) * np.array([-0.8, 0.4, 0.5, -0.6, 0.5, 0.9])
        leader[6] = leader[13] = 0.5 + 0.5 * np.sin(t)
        targets = ctrl.targets(leader)
        out = ctrl.step(leader)
        for side, sl in (("left", slice(0, 7)), ("right", slice(7, 14))):
            p, q = ctrl.solvers[side].fk(out[sl][:6])
            tp, tq = targets[side]
            pos_err.append(np.linalg.norm(p - tp))
            rot_err.append(quat_angle(q, tq))
        final = out
        if viewer is not None:
            viewer.sync()
            data.qpos[:] = out[0:6]
            mujoco.mj_forward(model, data)
            time.sleep(1.0 / args.rate)
    pos_err, rot_err = np.array(pos_err), np.array(rot_err)
    # Steady-state tracking (skip the first second while the IK catches up from the anchor).
    skip = int(args.rate) * 2
    p95, r95 = np.percentile(pos_err[skip:], 95), np.degrees(np.percentile(rot_err[skip:], 95))
    print(f"position error: median {np.median(pos_err[skip:])*1000:.2f} mm, p95 {p95*1000:.2f} mm, max {pos_err.max()*1000:.2f} mm")
    print(f"rotation error: median {np.degrees(np.median(rot_err[skip:])):.2f} deg, p95 {r95:.2f} deg")
    print("final command:", np.round(final, 3))

    # Static convergence: hold one target and iterate; FK(IK(target)) must converge to it.
    tgt = ctrl.targets(leader)
    for _ in range(int(args.rate)):
        out = ctrl.step(leader)
    p, q = ctrl.solvers["left"].fk(out[0:6])
    static_p = np.linalg.norm(p - tgt["left"][0])
    static_r = np.degrees(quat_angle(q, tgt["left"][1]))
    print(f"static convergence (1 s): {static_p*1000:.2f} mm, {static_r:.2f} deg")

    ok = static_p < 0.005 and static_r < 2.0 and p95 < 0.02
    print("PASS" if ok else "FAIL")
    if viewer is not None:
        viewer.close()
    return 0 if ok else 1


def hardware(args) -> int:
    from client.run_client import load_task_config
    from client.envs.dk1_env import _RealArms, _RealLeaders

    cfg = load_task_config(args.task_config)
    iv = dict(cfg.intervention)
    leaders = _RealLeaders(cfg.left_leader_port, cfg.right_leader_port,
                           cfg.leader_gripper_open_pos, cfg.leader_gripper_closed_pos)
    arms = _RealArms(cfg.left_follower_port, cfg.right_follower_port, control_mode=cfg.control_mode,
                     max_gripper_torque=cfg.max_gripper_torque, use_bimanual_class=cfg.use_bimanual_class)
    ik_keys = ("iterations", "position_cost", "orientation_cost", "gain", "max_velocity",
               "limit_margin", "limit_gain", "posture_cost", "max_joint_delta", "solver")
    ctrl = RelativeIKController(home14=cfg.reset_joints, dt=1.0 / iv.get("rate_hz", 60.0),
                                position_scale=iv.get("position_scale", 1.0),
                                rotation_scale=iv.get("rotation_scale", 1.0),
                                leader_joint_offsets=iv.get("leader_joint_offsets"),
                                leader_joint_signs=iv.get("leader_joint_signs"),
                                **{k: iv[k] for k in ik_keys if k in iv})
    try:
        print("Comparing FK(leader) with FK(follower) (follower model). Ctrl+C to continue.")
        try:
            while True:
                leader = ctrl.leader_to_follower_joints(leaders.get())
                follower = arms.get()
                parts = []
                for side, sl in (("L", slice(0, 7)), ("R", slice(7, 14))):
                    lp, lq = ctrl.solvers["left"].fk(leader[sl][:6])
                    fp, fq = ctrl.solvers["left"].fk(follower[sl][:6])
                    parts.append(f"{side}: dpos {np.linalg.norm(lp-fp)*100:5.1f} cm drot {np.degrees(quat_angle(lq,fq)):5.1f} deg "
                                 f"dq {np.round(leader[sl][:6]-follower[sl][:6],2)}")
                print(" | ".join(parts), flush=True)
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        input("\nPress Enter to anchor (dry run; nothing is sent).")
        ctrl.anchor(leaders.get(), arms.get())
        print("Anchored. Move the leaders. Ctrl+C to stop.")
        try:
            while True:
                t0 = time.perf_counter()
                out = ctrl.step(leaders.get())
                ms = (time.perf_counter() - t0) * 1000
                print(f"[{ms:5.1f} ms] target L {np.round(out[0:7],3)} R {np.round(out[7:14],3)}", flush=True)
                time.sleep(max(0.0, 1.0 / 60 - (time.perf_counter() - t0)))
        except KeyboardInterrupt:
            pass
    finally:
        leaders.close()
        input("Follower motors will be DISABLED next and the arms will fall: support both arms, then press Enter.")
        arms.close()
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hardware", action="store_true")
    p.add_argument("--task_config", default="configs/task/dk1_task.py")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--rate", type=float, default=60.0)
    args = p.parse_args()
    sys.exit(hardware(args) if args.hardware else synthetic(args))


if __name__ == "__main__":
    main()
