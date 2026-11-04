# EXPO-FT on a bimanual TRLC-DK1: implementation handover

Written 2026-11-03. Read this file in full before you change any code. These notes
come from a planning session that read the code and made no changes. Where a claim
here conflicts with the code, the code is right.

## Goal

Replicate EXPO-FT (the non-real-time pipeline, `scripts/pick/`) on a **bimanual
TRLC-DK1** robot in the user's lab this week. Lab work starts tomorrow (2026-11-04),
so all code must be written today and tested as far as possible without hardware.

User constraints:

- Keep changes to existing source small. Prefer new files and small hooks.
- Stick to the authors' libraries and pipeline where possible.
- Collect demos by teleoperating with the leader arms, convert them to LeRobot,
  and run supervised fine-tuning of pi0.5 before RL.
- Handle human interventions during RL with **anchored end-effector-relative
  motion from the leader arms, solved with IK**, as in amazon-far/abc. Do not use a
  SpaceMouse.

Settled decisions (do not reopen them):

| Topic | Decision |
|---|---|
| Pipeline | EXPO-FT (`EXPOLearner`, `--delay=0`) first. Use Real-Time EXPO-FT only if blocking inference is visibly a problem. |
| Action space | **14-D absolute joint positions**: `[L j1..j6, L grip, R j1..j6, R grip]`. Joints are in radians; the gripper runs from 0 (open) to 1 (closed). This is the DK1 LeRobot plugin's convention. State uses the same 14-D layout. |
| Delta actions | **Do not use them.** `Pi05.process_transformed_outputs` (`expo_ft/agents/vla/pi05.py:564`) runs the output transforms with a **zero state**, so openpi's `AbsoluteActions` would return deltas relative to 0. Set `use_delta_joint_actions=False`. |
| Base checkpoint | `gs://openpi-assets/checkpoints/pi05_base/params`, which the authors' DROID configs also use. Supervised fine-tuning is required because no pi0.5 checkpoint knows the DK1. |
| Dataset path | **Option C**: write our own collection loop on top of `DK1Env`, save the per-episode `traj.hdf5` layout the authors use, then convert to LeRobot with a generalized version of the authors' script. **Do not use `lerobot-record`.** It writes the LeRobot v3 format, and openpi (both upstream and the fork) pins an old LeRobot commit (`0cf8648…`, `lerobot.common.*` API, v2.x format). LeRobot ≥0.5 needs numpy ≥2, `datasets` ≥4 and Python ≥3.12. openpi needs numpy <2, and `pyproject.toml` overrides `datasets<4`. The two stacks cannot be installed together. Using the same `DK1Env` for demos and RL also keeps the observation pipeline identical. |
| openpi fork | Vendor it as a **git subtree**, not a clone and not a new branch of another repo. Details are in Step 1. |
| Success detection | Use the existing manual stdin detector (`success_detector_manual`: type 1 for success, 2 for reset) for the first task. |
| Remote | `origin` is now `https://github.com/sai-prasanna/expo-ft`. |

Ask the user before you start:

1. **Branch and commits.** May you commit the subtrees and the new code on a new
   branch such as `dk1`? Do not commit to `main` without asking. Push only if the
   user asks.
2. **First task.** Ask for the language prompt, a rough description of success,
   and the home pose.
3. **Hardware details,** if the user knows them: serial ports for the left and
   right follower and leader, camera device indices or serials and their names
   (assume `head`, `left_wrist`, `right_wrist`), whether a USB foot pedal is
   available (otherwise use a keyboard clutch), and the GPU server's hostname.
   Put any you don't get in the task config as clearly marked placeholders.

## How the existing code works (the parts that matter)

**Architecture.** The server runs the learner (`train_pi_robo.py`, JAX and openpi,
root `.venv`). The client runs the actor (`client/run_client.py`, a separate venv).
The client dials the server over a WebSocket and serves `create_env`, `reset`,
`step`, `get_observation`, `get_info_for_step` and `render`. The task config's
`env` class is constructed with `**task_config` as keyword arguments, plus
`video_dir` and `env_usage` (`run_client.py`, `create_env`). Every config key
therefore reaches the env constructor, so it must accept `**kwargs`.

**Env interface** (duck-typed; mirror `client/envs/droid_env.py:DroidEnv`):

- `reset()` returns an observation dict.
- `step(action)` returns a dict that contains `"executed_action"`.
- `get_observation()` returns an observation dict.
- `get_raw_observation()` and `transform_observation(raw)` are used by
  `collect_data.py`.
- `get_info_for_step(raw_obs=None)` returns `(done, success, reward, mask)`.
  The base class combines `success_detector_manual()`, `detect()` and the
  auto-reset timeout.
- Attributes: `control_hz` and `ignore_auto_reset`.

**Observation keys.** This is the most important convention. Observations must be
a **flat** dict of numpy arrays plus a `"prompt"` string. Two reasons:

- `Pi05.process_raw_inputs` (`pi05.py:538`) runs `np.asarray` on every top-level
  value and casts any key containing `"image"` to uint8. A nested `images` dict
  breaks this.
- Both the online path and the replay buffer apply the openpi config's
  **`repack_transforms` first**, then the data transforms
  (`pi05.py:461`, `replay_buffer.py:355`).

So the env observation keys **must equal the LeRobot dataset feature names** that
the repack transform maps from. The DROID example: the env emits
`exterior_image_1_left`, and the repack maps it to
`observation/exterior_image_1_left`. Suggested DK1 keys:
`head_image`, `left_wrist_image`, `right_wrist_image` (uint8 HxWx3, RGB), `state`
(float32, 14), and `prompt`. Use the same names in the LeRobot conversion and in
the openpi repack mapping.

**Demos for RL.** `train_pi_robo.py:128` (and the async script, and
`eval_droid_policy.py:69`) loads demos with `process_droid_dataset`
(`expo_ft/env/droid_utils.py`) **only when `env_type == "droid"`**. It reads each
`<dataset_path>/<int>/traj.hdf5` and walks the groups recursively. It takes
`saved_observation/*` as the observation and
`concat(action/<action_space>, action/gripper_<gripper_action_space>)` as the
action. The episode is labelled with reward 1 at the last step. It works with any
key names, so reuse it as is.

**Replay buffer.** It stores images under the slots `base_0_rgb`,
`left_wrist_0_rgb` and `right_wrist_0_rgb`, which match openpi's model image keys.
The critic cameras default to `("base_0_rgb", "left_wrist_0_rgb")`
(`expo_ft/agents/alg/batch_utils.py:12`). The task config can override this with
`critic_camera_keys` (read via `getattr` in `train_pi_robo.py:160`). Set all three
for bimanual.

**EXPO edits** happen in normalized model action space, so they don't depend on
the robot. `edit_action_xyzg` hard-codes DROID Cartesian slots 3 to 5
(`expo_ft.py:469`). Keep it `False`.

**Human intervention.**

- `run_client.py:_get_human_override_action` runs only when
  `env_type == "droid"`. It reads a SpaceMouse and treats the action as "human"
  when the motion norm exceeds 1e-4. The step handler then replaces the action and
  replies `action_type="human"`.
- The server (`train_pi_robo.py`, around line 360) stores
  `real_action, is_hil=True`, clears the plan, and **skips inference while
  `action_type == "human"`**. During that time it sends
  `np.zeros_like(example_action)`.
- The client skips `env.step` when the sent action is all `-1`
  (`sent_is_invalid`), unless a human is acting.

**The zero-action hazard.** The server sends zeros (a) at startup, through
`env.step(FLAGS.config_task.example_action)` (`train_pi_robo.py:310`,
`train_pi_robo_async.py:393`), and (b) for one step after every human hand-back,
through `np.zeros_like(example_action.squeeze())` (`train_pi_robo.py:362` and
`train_pi_robo_async.py:450`). Here `example_action` is the first demo's action,
not the config value, so only the `full_like` change fixes it. With velocity actions that is harmless.
**With absolute joint actions, zeros command every joint to 0 rad.** Fix it in
both places:

- Set `config.example_action = -np.ones((1, 14))` in the DK1 task config.
- Change both `zeros_like` lines to `np.full_like(..., -1.0)`. The client already
  ignores all-`-1` actions.
- As defense in depth, make `DK1Env.step` treat an all-`-1` action as "hold the
  current position".
- Check whether `eval_droid_policy.py` has an equivalent zero path; around line
  394 it only pops the plan.

## Work items

### Step 1: Repository plumbing

1. Remove `expo_ft/agents/vla/openpi/` from `.gitignore`, then vendor the openpi
   fork:

   ```bash
   git subtree add --prefix expo_ft/agents/vla/openpi https://github.com/pd-perry/openpi real-time-expo-ft --squash
   ```

   Use the **`real-time-expo-ft`** branch (head `2abe462`). It is strictly 3
   commits ahead of `expo_ft` (head `46407a4`); the extra commits touch
   `gemma.py`, `pi0.py`, `droid_policy.py` and `config.py`. This repo's `main`
   contains the real-time learners and prefix-conditioned code paths in
   `pi05.py`, so the superset branch is the safe choice for both pipelines.
   openpi's submodules (`third_party/aloha`, `third_party/libero`) are for
   examples only and are not needed. The root `pyproject.toml` already points
   `openpi` and `openpi-client` at this path.
2. Vendor trlc-dk1 the same way, because it may need patches after hardware
   testing:

   ```bash
   git subtree add --prefix third_party/trlc-dk1 https://github.com/robot-learning-co/trlc-dk1 main --squash
   ```

   (The commit that was read is `12f5368`.)
3. The DROID fork (`client/droid`) is **not** needed for the DK1. Leave the DROID
   client untouched.
4. Optional, and only if the user asks: the local `.git` is 1.2 GB because the
   remote branch `origin/view` (the project website) holds about 1.1 GB of MP4s.
   You could set the fetch refspec to `main` only, delete `origin/view`, and run
   `git gc --prune=now`.

### Step 2: DK1 client environment

Create `client_dk1/pyproject.toml` with `.python-version` set to 3.12 and
`package = false`, like `client/pyproject.toml`. Dependencies:

- `lerobot_robot_trlc_dk1` (path `../third_party/trlc-dk1`, editable), which pulls
  in `lerobot[dynamixel]>=0.5.1` and `mujoco`
- `mink==1.1.0` (the version abc uses) and `daqp`
- `openpi-client` (path `../expo_ft/agents/vla/openpi/packages/openpi-client`)
- `websockets~=13.1` (must match the server)
- `tyro`, `absl-py`, `ml-collections`, `h5py`, `imageio[ffmpeg]`, `numpy`

Run the scripts from the repo root as `python -m client.<module>` with this venv
active. Keep the `client` Python package shared; there is no need for a second
package. Do not import `droid` in any DK1 code path. Also make sure
`configs/task/*` imports degrade gracefully: they wrap env imports in
`try`/`except`.

### Step 3: `client/envs/dk1_env.py` (`DK1Env`)

Facts about trlc-dk1 (read `third_party/trlc-dk1/lerobot_robot_trlc_dk1/*.py`,
`trlc_dk1_control/*.py`, `examples/bi_teleop.py` and `README.md`):

- Plugin classes:
  - `dk1_follower` / `DK1Follower`, config fields `port`, `control_mode`
    (default `"impedance"`, or `"pos_vel"`), `max_gripper_torque`, `cameras`,
    `joint_velocity_scaling`
  - `dk1_leader` / `DK1Leader`, config fields `port`, `gripper_open_pos`,
    `gripper_closed_pos`
  - `bi_dk1_follower` / `BiDK1Follower`, config fields `left_arm_port`,
    `right_arm_port`, `cameras`
  - `bi_dk1_leader` / `BiDK1Leader`
- Feature keys are `joint_1.pos` … `joint_6.pos` and `gripper.pos`. The
  bimanual classes prefix them with `left_` and `right_`. Actions use the same
  keys and are absolute positions.
- Follower motors are DAMIAO DM4340/DM4310 on CAN-over-serial. The impedance
  controller runs a 250 Hz MIT-mode thread with MuJoCo gravity compensation.
  Leader motors are Dynamixel xl330 (7 servos: 6 joints and a trigger).
- Cameras use LeRobot `OpenCVCameraConfig`. The README example is 960x540 at
  60 fps.
- **Possible bug (unconfirmed; test on hardware first):** `BiDK1Follower._motors_ft`
  reads `self.left_arm.motors`, but `DK1Follower` seems to have only `_motors`.
  Also, `BiDK1FollowerConfig` doesn't forward `control_mode`. If the bimanual
  class fails, drive two `DK1Follower`s (and two `DK1Leader`s) directly. Write
  `DK1Env` so it can use either.

Requirements for `DK1Env`:

- Constructor takes `**kwargs` (the task config is passed in whole). Connect the
  follower arms and cameras here, using LeRobot camera classes.
- `get_raw_observation()` reads the follower state and cameras.
  `transform_observation()` resizes to `config.image_size` (keep the authors'
  `(180, 320)` or use 224x224), converts to RGB uint8, and returns the flat dict
  described above. `get_observation()` returns the transformed dict.
- `step(action)` takes the 14-D action. An all-`-1` action means hold. Otherwise:
  - clip to the joint limits (`trlc_dk1_control/config.py`) and the gripper to
    [0, 1];
  - **clip the per-step joint change** to a value in the config
    (`max_joint_delta_per_step`, for example 0.05 rad at 15 Hz);
  - optionally enforce a box on the end-effector position using FK from the IK
    module;
  - send the action and return `{"executed_action": clipped_action}`.
- `reset()` interpolates slowly to `config.reset_joints` (14-D, blocking), opens
  the grippers, waits for the user (manual scene reset; for example, print and
  wait for Enter when `config.wait_for_scene_reset` is set), resets the
  detector, and returns an observation.
- `get_info_for_step()` copies the `DroidEnv` logic: manual detector first, then
  `detect()` (returns `(False, False)` by default), then the timeout
  (`auto_reset_steps`, skipped when `ignore_auto_reset`).
- `get_human_action()` returns `(action14, is_human)` from the intervention
  controller (Step 4). The controller is created lazily so plain rollouts don't
  need the leader arms.
- Provide `close()`, and optionally `render()` (head camera).
- Provide a **`mock=True` mode**: fake arms that integrate the commanded joints
  and noise images. This lets you test everything end to end today without
  hardware.

### Step 4: Intervention by anchored relative IK

Port the scheme from amazon-far/abc (commit `d0832d1`), files
`deploy/robot/gym/dagger_rollout.py` (`RelativeGelloController`, `DaggerLoop`),
`deploy/robot/control/relative_pose.py` and `deploy/robot/control/mink_ik.py`.
Put it in a new `client/teleop/` package (`relative_ik.py`, `mink_ik.py`,
`clutch.py`). Fetch abc to read it; do not vendor abc.

How abc does it:

- **On clutch-in (anchor),** for each arm:
  - `anchor_leader = FK(leader_joints)` (using the **follower's** model);
  - `anchor_follower = FK(follower_joints)`;
  - record a gripper offset.
- **Each step:**

  ```
  target_pos = anchor_follower_pos + (leader_pos - anchor_leader_pos) * pos_scale
  target_rot = scale(q_leader ⊗ q_anchor_leader⁻¹, rot_scale) ⊗ q_anchor_follower
  ```

  The rotation delta is applied in the world frame (left-multiplied). Then
  `joints = IK(target, seed=previous_command)`.
- **IK** uses `mink` with MuJoCo: a `FrameTask` on the end-effector frame
  (position cost 50, orientation cost 5), a `PostureTask` toward mid-range
  joints, `ConfigurationLimit`, and `VelocityLimit` at 3 rad/s. The solver is
  daqp, 4 iterations per call, `dt=1/60`. Each call clips the joint change to
  0.12 rad and returns the seed if no solution is found. Use one solver per arm.
- **Gripper:** abc uses an offset with snap rules. Mapping the leader's trigger
  value (0 to 1) directly is fine.
- **Switching is a hard switch** between policy and intervention. There is no
  blending.

DK1 specifics and gotchas:

- Use the URDF at `third_party/trlc-dk1/urdf/follower/TRLC-DK1-Follower.urdf`.
  Joints are `joint1` to `joint6` (no underscore, unlike the LeRobot keys), plus
  prismatic `gripper_left` and `gripper_right`. The end-effector frame is the
  fixed `tool0` (0.158 m from `link6-7`). **MuJoCo's URDF import merges
  fixed-joint bodies by default, so `tool0` disappears.** Either add
  `<mujoco><compiler fusestatic="false" meshdir="..."/></mujoco>` to a copy of
  the URDF, or add a site to the last link at the `tool0` offset. See how
  `trlc_dk1_control/gravity_comp.py` loads the URDF. Exclude or lock the gripper
  joints in the IK model.
- abc runs FK on the **leader joints with the follower model**. That assumes the
  GELLO-style leader matches the follower's kinematics and zero offsets. **Check
  this:** with teleop mirroring, FK(leader) should be close to FK(follower). If it
  isn't, build a leader joint → follower joint offset/sign map. The leader's
  conversion is `val / 4096 * 2π − π`; see `leader.py`.
- **Rate:** `run_client` only sees `step` at `control_hz`. Run the leader read
  and IK in a **background thread at about 60 Hz** that keeps the latest 14-D
  target. `get_human_action()` returns that target. This keeps the IK smooth and
  stops the IK seed from going stale.
- **Clutch:** use a key or foot-pedal toggle, not motion detection. Engaging
  re-anchors; disengaging hands control back to the policy. A keyboard fallback
  can use `pynput`, which comes with LeRobot, or a `/dev/tty` reader thread.
  **Avoid the digits 1 and 2:** `success_detector_manual` reads lines from
  `/dev/tty`, so use a different mechanism or key set.
- **Wire it in** with about 10 lines in `run_client.py`, in the `step` handler:
  if `hasattr(env, "get_human_action")`, call it instead of the SpaceMouse. If a
  human is acting, set `real_action = action` and `action_type = "human"`. Keep
  the DROID path unchanged.
- Add a standalone test script (`client/teleop/test_relative_ik.py`). With `mock`
  set, it runs the IK on recorded or synthetic leader motion in MuJoCo and
  optionally shows it in the MuJoCo viewer. On hardware, it does a dry run that
  prints targets without sending them.

### Step 5: Data collection (`client/collect_data_dk1.py`)

Copy the structure of `client/collect_data.py`: absl flags `save_root`,
`num_episodes`, `task_config` and `test_detector`; temp-dir routing; only
successes go to `<save_root>/success/<id>/`; streaming MP4 previews. Changes:

- The controller **mirrors the leader joints directly** onto the follower. This is
  plain joint-space teleop, not the relative IK, because that's best for demos and
  is what `lerobot-record` does.
- Recording starts on the first leader motion or a key press.
- **Replace DROID's `TrajectoryWriter`** with a small h5py writer: append-only
  datasets per key, chunked. The layout must match what `process_droid_dataset`
  and the conversion script expect:
  - `saved_observation/<obs_key>`, one per flat observation key (images as
    uint8). Save `prompt` as a fixed string or leave it out; `process_droid_dataset`
    decodes bytes.
  - `action/action14`: one 14-D dataset in the `[L6, Lg, R6, Rg]` layout, with
    `config.action_space="action14"` and `config.gripper_action_space=None`.
    Don't split it into `joint_position` and `gripper_position`:
    `process_droid_dataset` concatenates arm then gripper, which would give
    `[L6, R6, Lg, Rg]`. Make a small, backward-compatible change to
    `process_droid_dataset` so it skips the gripper concatenation when
    `gripper_action_space` is `None`.
  - The stored action must be **what was commanded**, the clipped
    `executed_action`, in the same 14-D layout as the policy action.
- Remove the `env.camera_reader` call, which is DROID-only. Keep the timing prints.
- Before collecting, **record the actual loop rate**: the camera reads plus the
  leader read must fit in `1/control_hz`.

### Step 6: Server-side changes

1. **openpi subtree.** Add `src/openpi/policies/dk1_policy.py` with `DK1Inputs`
   and `DK1Outputs`, modelled on `aloha_policy.py` but with **no `adapt_to_pi`
   joint flips or gripper conversion**:
   - Inputs: `state` is 14-D, padded to `model.action_dim=32`. Images map to
     `base_0_rgb` (head), `left_wrist_0_rgb` and `right_wrist_0_rgb`, all masks
     True. Pass the prompt through.
   - Outputs: `actions[:, :14]`.

   Then add a `LeRobotDK1DataConfig` (a `DataConfigFactory`) to
   `src/openpi/training/config.py`, with a repack that maps the dataset keys to
   themselves, `DK1Inputs`/`DK1Outputs`, **no `DeltaActions`**, and the standard
   model transforms. Copy the DROID config's details, such as how it builds
   `ModelTransformFactory` and its image resize option. Finally add
   `TrainConfig` entries `expo_pi05_dk1_lora_finetune_sft` (and a full fine-tune
   variant if it's cheap), copied from
   `expo_pi05_droid_lora_finetune_sft_cartesian_state`:
   `Pi0Config(pi05=True, action_dim=32, action_horizon=16,
   paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")`,
   the same freeze filter and EMA, and weights from `pi05_base`. Check whether the
   real-time branch's changes to `config.py`/`droid_policy.py` need DK1
   equivalents, such as per-token conditioning flags.
2. **Gates.** In `train_pi_robo.py`, `train_pi_robo_async.py` and
   `eval_droid_policy.py`, accept `env_type in ("droid", "dk1")` and use the
   (generalized) `process_droid_dataset` for both.
3. **Zero-action fix**, as described in "The zero-action hazard" above.
4. **Conversion.** Generalize `scripts/convert_droid_data_to_lerobot.py`, or add
   `scripts/convert_dk1_data_to_lerobot.py`. It reads `traj.hdf5`
   (`saved_observation` images and state, plus the 14-D action) and writes LeRobot
   features with the **same names** as the env observation keys. Images are type
   `image` (the authors' style), `state` has shape (14,), `actions` has shape
   (14,), the fps is `control_hz`, and the task comes from
   `config.language_instruction`. It runs in the **server** venv (old LeRobot API,
   `lerobot.common.datasets…`). Watch for the hard-coded
   `action_dim = 7 if action_key == "cartesian_velocity" else 8`.
5. **Task configs.**
   - `configs/task/dk1_base.py`: `env_type="dk1"`, `env=DK1Env`, ports and
     cameras as placeholders, `image_size`, `control_hz` (start at 15),
     `reset_joints` (14), `example_action=-ones((1, 14))`,
     `action_space="action14"`, `gripper_action_space=None`,
     `edit_action_xyzg=False`,
     `critic_camera_keys=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")`,
     `auto_reset_steps`, `max_joint_delta_per_step`, IK and clutch settings,
     `mock=False`.
   - `configs/task/dk1_<task>.py` inherits from it, the way `dynamic_pick.py`
     inherits from `pick.py`.
   - Check every task config key is safe to pass as a constructor keyword.
6. **Scripts.** Create `scripts/dk1_<task>/`, copying `scripts/pick/`:
   `collect_data.sh` (uses the client_dk1 venv), `convert_data.sh`,
   `calculate_norm.sh`, `finetune.sh`, `run_policy.sh`, `run_server.sh`,
   `run_server_async.sh` and `eval_policy.sh`, with the new config and asset
   names. Keep the authors' hyperparameters (4000 SFT steps, `N=8`,
   `n_edit_samples=8`, `edit_scale=0.2`, `update_type=episode`, `num_updates=3`,
   `replan_steps=8`) unless there's a reason to change them.

### Step 7: Verify without hardware (must pass before the lab)

1. `uv sync` in the root and in `client_dk1/`.
2. In the server venv, check
   `python -c "from openpi.training import config; config.get_config('expo_pi05_dk1_lora_finetune_sft')"`.
3. Run mock collection: `collect_data_dk1.py` with `mock=True` (scripted or
   automatic success) for 3 episodes to produce `traj.hdf5` files.
4. Convert them to LeRobot and compute norm stats with openpi's
   `scripts/compute_norm_stats.py --config-name ... --repo_id ...`.
5. Check that `process_droid_dataset` loads the mock episodes. Also push one
   observation through `Pi05.process_raw_inputs` and through replay-buffer
   insertion (`_preprocess_single_transition`) with that config, and confirm the
   shapes: state 32, images 224, actions (16, 32).
6. If a GPU is available, run a few SFT steps (`--num_train_steps=10`). Then run
   a short `train_pi_robo.py` against the mock client over localhost to cover the
   handshake, step, the intervention path and the `-1` hold logic.
7. Run the IK test in MuJoCo with a synthetic anchored motion and confirm that
   FK(IK(target)) is close to the target.

## Plan for the lab

1. Run the trlc-dk1 examples (`teleop.py`, `bi_teleop.py`) to check the ports,
   calibration and the bimanual class. Patch the subtree if needed.
2. Run `DK1Env` reset and step, and check the clipping and hold behavior.
3. Collect about 20 demos with `collect_data_dk1.py`.
4. Convert, compute norm stats, and run SFT (4000 steps) on the server.
5. Evaluate for non-zero success.
6. Check that FK(leader) matches FK(follower) and test the relative-IK takeover
   on its own.
7. Run online EXPO-FT with interventions.

## External references

| Repo | Commit read | Notes |
|---|---|---|
| github.com/robot-learning-co/trlc-dk1 | `12f5368` | LeRobot plugin `lerobot_robot_trlc_dk1`; needs Python ≥3.12 and `lerobot[dynamixel,viz]>=0.5.1`. |
| github.com/pd-perry/openpi | `real-time-expo-ft` `2abe462`, `expo_ft` `46407a4` | Upstream openpi (`main`, 2026-08-24) still pins the same old LeRobot. Existing bimanual references in the fork: `aloha_policy.py`, `pi05_aloha`, `pi05_aloha_pen_uncap`. The ALOHA layout is the same 14-D order. |
| github.com/amazon-far/abc | `d0832d1` | See `deploy/README.md`. YAM-specific parts (`i2rt` drivers, `yam.xml`, CAN scripts) are not needed. |
| Original expo-ft README | | Workflow; `scripts/pick/*.sh` for hyperparameters. |

## Implementation status (2026-11-03)

Implemented on branch `dk1`. Verified without hardware:

- `uv sync` in the root and in `client_dk1/` (the client venv overrides `numpy>=2.0.1`,
  because `openpi-client` pins `numpy<2` and lerobot 0.6 needs numpy 2).
- `config.get_config('expo_pi05_dk1_lora_finetune_sft')` loads.
- Mock collection (`configs/task/dk1_mock.py`) produced 3 episodes, which converted to
  LeRobot; norm stats (14-D) computed.
- `process_droid_dataset` loads the mock demos. Replay buffer and online transforms give
  state (32,), images 224x224 (all 3 masks True), action chunks (16, 32).
- 10 SFT steps with `debug_pi05_dk1` (tiny model, DK1 transforms).
- `train_pi_robo.py` against the mock client over localhost: handshake, startup `-1` hold,
  10 rollout episodes, inference and buffer insertion, clutch engage/release. The first
  gradient update does not fit on a 16 GB GPU even with the debug model (SigLIP is full
  size), so updates need the lab GPU server.
- Client step handler: `-1` is skipped, human override returns `action_type=human`, and
  large jumps are clipped.
- Relative IK in MuJoCo: steady-state tracking p95 below 0.1 mm; static FK(IK(target)) converges.

Not verified: anything that needs hardware or the `pi05_base` download.

Additions beyond the plan:

- `client/dk1_setup_hardware.py` writes `configs/task/dk1_hardware.json` (ports by
  unplug/replug, cameras by preview, `--check`). `dk1_base.py` reads it.
- Clutch: the default is a second terminal (`python -m client.teleop.clutch`, SPACE
  toggles, UDP), because pynput does not see Wayland terminals. A bare Enter in the main
  terminal blocks `success_detector_manual`, so do not use it as a key.
- One per-step joint cap for collection and RL (`max_joint_delta_per_step`, 0.25 rad, measured from the follower pose), with 100 Hz command smoothing on.
- `run_client` sends a hold (all `-1`) instead of zeros for NaN actions on DK1.
- `BiDK1Follower` bug confirmed: `_motors_ft` uses `.motors`, but it is only reached through
  `observation_features`. `DK1Env` drives two `DK1Follower`s by default
  (`use_bimanual_class=False`).
