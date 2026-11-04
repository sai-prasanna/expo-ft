"""Offline sanity check of a DK1 SFT checkpoint on recorded demos. No robot is touched.

For frames every `stride` steps of a few demo episodes, runs the policy (standard openpi
inference: DK1Inputs -> normalize -> model -> unnormalize -> DK1Outputs) and compares the
first `replan` predicted actions with the recorded actions. Also reports how far the first
predicted action is from the current state (what the env's per-step joint cap sees), next to
the same quantity for the human demos.

  .venv/bin/python scripts/dk1_task/check_policy_offline.py \
      --ckpt checkpoints/expo_pi05_dk1_lora_finetune_sft/dk1_task_20_lora_sft/4000
"""

import argparse
import dataclasses
import os

import h5py
import numpy as np

from openpi.policies import policy_config
from openpi.training import config as _config

JOINTS = [f"L{i}" for i in range(1, 7)] + ["Lg"] + [f"R{i}" for i in range(1, 7)] + ["Rg"]
ARM = np.r_[0:6, 7:13]  # arm joints (no grippers)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default="expo_pi05_dk1_lora_finetune_sft")
    p.add_argument("--repo_id", default="expo_ft/dk1_task_20")
    p.add_argument("--data", default="data/stack_green_blue/success")
    p.add_argument("--episodes", type=int, nargs="*", default=[0, 7, 15])
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--replan", type=int, default=8)
    args = p.parse_args()

    cfg = _config.get_config(args.config)
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(
        cfg.data, repo_id=args.repo_id,
        assets=dataclasses.replace(cfg.data.assets, asset_id=args.repo_id)))
    policy = policy_config.create_trained_policy(cfg, args.ckpt)

    errs, jumps_pol, steps_pol = [], [], []
    for ep in args.episodes:
        with h5py.File(os.path.join(args.data, str(ep), "traj.hdf5"), "r") as f:
            so = f["saved_observation"]
            T = so["state"].shape[0]
            act = f["action"]["action14"][:]
            for t in range(0, T - args.replan, args.stride):
                prompt = so["prompt"][t]
                obs = {
                    "head_image": so["head_image"][t],
                    "left_wrist_image": so["left_wrist_image"][t],
                    "right_wrist_image": so["right_wrist_image"][t],
                    "state": so["state"][t].astype(np.float32),
                    "prompt": prompt.decode() if isinstance(prompt, bytes) else str(prompt),
                }
                chunk = np.asarray(policy.infer(obs)["actions"])[: args.replan]
                errs.append(np.abs(chunk - act[t : t + args.replan]))
                jumps_pol.append(np.abs(chunk[0] - obs["state"]))
                steps_pol.append(np.abs(np.diff(chunk, axis=0)))
            demo_jump = np.abs(act - so["state"][:])
            demo_step = np.abs(np.diff(act, axis=0))
        print(f"ep {ep}: T={T}")

    errs = np.concatenate(errs)  # (n*replan, 14)
    jumps_pol, steps_pol = np.stack(jumps_pol), np.concatenate(steps_pol)
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    print("\njoint                 ", "  ".join(f"{j:>6}" for j in JOINTS))
    print("MAE pred vs demo (rad)", "  ".join(f"{v:6.3f}" for v in errs.mean(0)))
    print("max pred vs demo      ", "  ".join(f"{v:6.3f}" for v in errs.max(0)))
    print("|a0 - state| policy p95", "  ".join(f"{v:6.3f}" for v in np.percentile(jumps_pol, 95, axis=0)))
    print("|a  - state| demo   p95", "  ".join(f"{v:6.3f}" for v in np.percentile(demo_jump, 95, axis=0)))
    print("|a_t+1 - a_t| policy max", "  ".join(f"{v:6.3f}" for v in steps_pol.max(0)))
    print("|a_t+1 - a_t| demo   p99", "  ".join(f"{v:6.3f}" for v in np.percentile(demo_step, 99, axis=0)))
    print(f"\narm joints: policy |a0-state| max={jumps_pol[:, ARM].max():.3f} rad; "
          f"steps over 0.25 rad: {(jumps_pol[:, ARM] > 0.25).any(1).mean() * 100:.0f}%")


if __name__ == "__main__":
    main()
