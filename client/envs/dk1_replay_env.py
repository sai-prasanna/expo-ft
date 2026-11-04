"""Hardware-free DK1 env that serves recorded demo frames (traj.hdf5) to the policy server.

Each reset starts the next demo episode; get_observation returns that demo's frame t, and step
records the policy's action next to the demo's recorded action, then advances t. The policy
never moves anything, so this checks the *exact* server pipeline (eval_droid_policy.py /
train_pi_robo.py inference path) on real images before a checkpoint touches the robot.

Per episode it prints, for the 12 arm joints:
  err     |a - demo action|                      (how far from what the human did)
  jump    |a - current state|                    (what the real env's per-step cap sees)
  over    % of steps with any arm jump > max_joint_delta_per_step
  step    max |a_t+1 - a_t|                      (discontinuities, e.g. at chunk boundaries)
"""

import glob
import json
import os

import h5py
import numpy as np

ARM = np.r_[0:6, 7:13]


class DK1ReplayEnv:
    def __init__(self, replay_data_dir="data/stack_green_blue/success", replay_episodes=None,
                 replay_log="", max_joint_delta_per_step=0.25, language_instruction="", **_):
        paths = sorted(glob.glob(os.path.join(replay_data_dir, "*", "traj.hdf5")),
                       key=lambda p: int(os.path.basename(os.path.dirname(p))))
        if replay_episodes:
            paths = [p for p in paths if int(os.path.basename(os.path.dirname(p))) in replay_episodes]
        if not paths:
            raise FileNotFoundError(f"no traj.hdf5 under {replay_data_dir}")
        self.paths, self.ep_idx = paths, -1
        self.cap = float(max_joint_delta_per_step or 0.25)
        self.prompt = language_instruction
        self.log_path = replay_log
        self.results = []
        self._ep = None

    def _load(self, path):
        with h5py.File(path, "r") as f:
            so = f["saved_observation"]
            return {
                "head_image": so["head_image"][:], "left_wrist_image": so["left_wrist_image"][:],
                "right_wrist_image": so["right_wrist_image"][:], "state": so["state"][:].astype(np.float32),
                "action": f["action"]["action14"][:].astype(np.float32), "path": path,
            }

    def _summarize(self):
        if self._ep is None or not self._acts:
            return
        a = np.asarray(self._acts)
        n = len(a)
        demo, state = self._ep["action"][:n], self._ep["state"][:n]
        err = np.abs(a - demo)[:, ARM]
        jump = np.abs(a - state)[:, ARM]
        steps = np.abs(np.diff(a, axis=0))[:, ARM] if n > 1 else np.zeros((1, 12))
        demo_jump = np.abs(demo - state)[:, ARM]
        r = {
            "episode": os.path.basename(os.path.dirname(self._ep["path"])), "steps": n,
            "err_mean": float(err.mean()), "err_max": float(err.max()),
            "jump_max": float(jump.max()), "demo_jump_max": float(demo_jump.max()),
            "over_cap_pct": float((jump > self.cap).any(1).mean() * 100),
            "step_max": float(steps.max()), "demo_step_max": float(np.abs(np.diff(demo, axis=0))[:, ARM].max()),
        }
        self.results.append(r)
        print("[replay] " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()),
              flush=True)
        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(r) + "\n")

    def reset(self):
        self._summarize()
        self.ep_idx = (self.ep_idx + 1) % len(self.paths)
        self._ep = self._load(self.paths[self.ep_idx])
        self.t, self._acts = 0, []
        return self.get_observation()

    def get_observation(self):
        t = min(self.t, len(self._ep["state"]) - 1)
        obs = {k: self._ep[k][t] for k in ("head_image", "left_wrist_image", "right_wrist_image", "state")}
        obs["prompt"] = self.prompt
        return obs

    def step(self, action, max_delta=None):
        a = np.asarray(action, dtype=np.float32).reshape(-1)[:14]
        hold = np.allclose(a, -1.0)
        if not hold:
            self._acts.append(a)
            self.t += 1
        return {"executed_action": a if not hold else self._ep["state"][min(self.t, len(self._ep["state"]) - 1)]}

    def get_info_for_step(self, raw_obs=None):
        done = self.t >= len(self._ep["state"]) - 1
        if done:
            self._summarize()
            self._acts = []
        return done, False, 0.0, 1.0

    def render(self):
        return self._ep["head_image"][min(self.t, len(self._ep["state"]) - 1)]

    def close(self):
        self._summarize()
