# DK1 lab runbook

Follow this from top to bottom to set up the bimanual TRLC-DK1, check it, collect
demos, fine-tune pi0.5 and run EXPO-FT. Each stage ends with a check. Don't move on
until the check passes.

Two machines are involved:

- **Robot PC**: the arms and cameras are plugged in here. It uses the `client_dk1`
  virtual environment.
- **GPU server**: it runs SFT, evaluation and the EXPO-FT learner, using the root `.venv`.
  It needs a GPU with more than 22.5 GB of memory. A 16 GB card can't run LoRA SFT or the
  EXPO update. The two machines can be the same one if it has such a GPU.

Run every command from the repository root.

Keep a hand near the arms whenever the motors are enabled. **Disconnecting a follower
disables its motors and the arm falls.** Every script asks you to hold the arms before
that happens (or moves to `rest_joints` if you set it).

## Install

1. On both machines, get the branch:

   ```bash
   git fetch origin && git checkout dk1 && git pull
   ```

2. On the robot PC, install the client environment:

   ```bash
   cd client_dk1 && uv sync && cd ..
   source client_dk1/.venv/bin/activate
   ```

3. On the GPU server, install the server environment:

   ```bash
   GIT_LFS_SKIP_SMUDGE=1 uv sync
   ```

4. On the robot PC, make sure your user can open the serial and video devices:

   ```bash
   groups | grep -E "dialout|video"
   ```

   If a group is missing, run `sudo usermod -aG dialout,video $USER`, then log out and
   back in.

**Check:** `client_dk1/.venv/bin/python -m client.teleop.test_relative_ik` prints `PASS`.

## Configure the hardware

The setup script writes `configs/task/dk1_hardware.json`, which every DK1 script reads.

1. Plug in and power both followers, both leaders and the three cameras.
2. To find the four serial ports, run the setup script and follow its prompts:

   ```bash
   python -m client.dk1_setup_hardware
   ```

   a. For each arm, the script asks you to unplug its USB cable, then plug it back in.
      This identifies left and right without guessing.
   b. For each camera, it saves a preview to `outputs/dk1_camera_previews/` and asks
      for its role: `head`, `left_wrist`, `right_wrist`, or `skip`.

3. Open `configs/task/dk1_hardware.json` and correct anything that is wrong:
   - To swap left and right, swap the port values.
   - To rotate a camera image, add `"rotation": 180` to that camera.
   - If three cameras at once drop frames, add `"fourcc": "MJPG"` to each camera, or lower
     the resolution.
4. To connect to every device once, run:

   ```bash
   python -m client.dk1_setup_hardware --check
   ```

**Check:** the command prints three image shapes, a 14-value follower state and a
14-value leader state, then `OK`.

If the followers fail to connect, run the vendor's single-arm example to check the
motors and calibration on their own:
`third_party/trlc-dk1/examples/follower_read_position.py`. Edit the port at the top of
the file first.

## Home pose

Nothing to do. The home pose is where every episode starts, and where the arms park
before the motors are disabled on close. It is the DK1 folded pose: all joints at `0`
(the calibration zero), grippers open (`DK1_FOLDED_POSE` in `configs/task/dk1_base.py`).
To override it, add a `"reset_joints"` (or, for parking only, `"rest_joints"`) list to
the JSON file with 14 values in the `[L j1..j6, L grip, R j1..j6, R grip]` order.

Commit the hardware file:

```bash
git add configs/task/dk1_hardware.json && git commit -m "DK1 lab hardware config"
```

## Set the task

In `configs/task/dk1_task.py`, replace the placeholders:

- Set `config.language_instruction` to the task prompt. The prompt is written into the
  LeRobot dataset, so set it before you convert any demos.
- Note in the comment what counts as success, so every operator labels episodes the
  same way.

## Check the leader-to-follower kinematics

Interventions apply the leader's end-effector motion to the follower. That requires the
leader joints to follow the same convention as the follower joints.

1. Run the dry run. It sends nothing to the followers:

   ```bash
   python -m client.teleop.test_relative_ik --hardware --task_config configs/task/dk1_task.py
   ```

2. Line up each leader with its follower by eye; the follower motors hold their pose.
   Watch the printed `dpos`, `drot` and per-joint `dq`.
3. If `dpos` is more than about 2 cm or a joint shows a constant offset or a flipped sign,
   set `leader_joint_offsets` or `leader_joint_signs` under `config.intervention` in
   `configs/task/dk1_base.py`. Rerun until the values are small.
4. Press **Ctrl+C**, then **Enter** to anchor. Move the leaders and check that the printed
   joint targets change smoothly. Press **Ctrl+C** to stop.

**Check:** `dpos` is under about 2 cm and `drot` is under about 5 degrees when the arms
match.

## Do a collection dry run

A dry run tests reset, teleoperation, the manual success prompt and the loop rate,
without saving anything:

```bash
python -m client.collect_data_dk1 --task_config configs/task/dk1_task.py --test_detector
```

1. The arms move to home. Reset the scene and press **Enter**.
2. The followers ease onto the leaders. Move the leaders around.
3. To end an episode, type `1` and **Enter** (success) or `2` and **Enter** (discard).
   Never press a bare **Enter** during an episode: the success prompt then waits for input
   and blocks the loop.
4. Watch the `timing ms:` lines. `work` must stay under the 66.7 ms budget at 15 Hz.
5. At each reset, read the `per-step joint cap hit on N/M steps` line. If it's high at your
   normal teleoperation speed, raise `max_joint_delta_per_step` in `dk1_base.py`.
6. Motion should be smooth: `config.command_interp_hz = 100.0` (on by default) ramps each 15 Hz
   target. If it still jerks, check the driver gains; if it feels held back, check the cap-hit line.
7. Press **Ctrl+C** to stop, then hold the arms when prompted.

**Check:** episodes start and end as expected, `work` fits the budget, and motion is
acceptable.

## Collect demos

Collect about 20 successful demos:

```bash
bash scripts/dk1_task/collect_data.sh
```

Episodes are hands-free by default (`--gesture_control`; `--nogesture_control` restores the
keyboard-only flow). Only the first episode homes and waits for **Enter**. After that, teleop runs
continuously, and every episode goes home → task → home:

1. **Idle.** Nothing is recorded. Reset the scene; take as long as you need. It becomes ready
   once the leaders are at home, still, with the triggers open (`[gesture] Ready`).
2. **Start.** Clearly move an arm (any joint more than 0.1 rad): recording starts. The 0.5 s
   of rest just before the motion is kept, so every episode starts at rest. Slow sag while you
   are away and the triggers never start an episode.
3. **Stop.** Bring the leaders back home and hold still for 0.5 s. "Home" is loose: only the
   shoulder/elbow joints (j1–j3) count, within 0.4 rad of `reset_joints`; wrists can be
   anywhere. It can only stop after the arms have gone 0.6 rad from home, so working close to
   home never cuts an episode short.
4. **Decide.** It waits for you. Squeeze (above 0.5) and hold for 0.6 s: **both triggers =
   keep**, **left trigger alone = discard** (right alone does nothing). Then back to 1.

`1`/`2` + **Enter** still end an episode. Nothing is ever dropped without your gesture. Each
`traj.hdf5` stores how it ended in the `end_reason` attribute. A `[gesture …]` status line
(every 0.5 s; every 5 s while idle) shows the distance from home with the joint that's furthest
off, the joint speed, and whether each check passes (`ok`/`X`). The radii and timings are
flags (`--gesture_*`, see `client/collect_data_dk1.py`).

Only successful episodes are kept, in `data/dk1_task/success/<id>/traj.hdf5`, with MP4
previews next to them in `recordings/MP4/`. Running the script again with the same folder
adds episodes; it never overwrites them.

To collect for several tasks, use one folder and one prompt per task:

```bash
SAVE_ROOT=data/dk1_pick PROMPT="pick up the cube" bash scripts/dk1_task/collect_data.sh
```

The prompt is stored with every episode. The LeRobot conversion uses each episode's stored
prompt, so you can point `DATA_DIR` in `convert_data.sh` at `data/` to fine-tune one
multi-task policy. For EXPO-FT, set `language_instruction` in the task config to that task's
prompt and point `--dataset_path` at that task's `success/` folder.

**Check:** `ls data/dk1_task/success` shows 20 numbered folders. Open one or two MP4
previews: all three cameras should be visible, with the correct left and right.

## Fine-tune pi0.5 on the GPU server

1. If the GPU server is a different machine, copy the demos to it:

   ```bash
   rsync -av data/dk1_task/ GPU_SERVER:/path/to/expo-ft/data/dk1_task/
   ```

2. On the GPU server, convert the demos to LeRobot and compute the normalization stats:

   ```bash
   bash scripts/dk1_task/convert_data.sh
   bash scripts/dk1_task/calculate_norm.sh
   ```

   The dataset name is `expo_ft/dk1_task_20`. If you collected a different number of
   demos, change `MAX_EPISODES` / `REPO_ID` / `DATA_ID` in all `scripts/dk1_task/*.sh`.
3. Run 4000 LoRA SFT steps. The first run downloads the `pi05_base` weights:

   ```bash
   bash scripts/dk1_task/finetune.sh
   ```

**Check:** the loss decreases, and
`checkpoints/expo_pi05_dk1_lora_finetune_sft/dk1_task_20_lora_sft/4000/params` exists.

## Evaluate the SFT policy

The policy must succeed at least sometimes before RL is worth running.

1. On the GPU server, start the evaluator:

   ```bash
   bash scripts/dk1_task/eval_policy.sh
   ```

2. On the robot PC, connect the client:

   ```bash
   SERVER_HOST=GPU_SERVER_HOSTNAME bash scripts/dk1_task/run_policy.sh
   ```

   If both run on the same machine, leave out `SERVER_HOST`; it defaults to `localhost`.
3. For each episode, reset the scene, press **Enter**, then type `1` or `2` and **Enter**
   to label the outcome. Episodes also end after `auto_reset_steps` (300 steps, 20 s).

**Check:** the arms move toward the task, there are no sudden jumps, and some episodes
succeed.

## Run EXPO-FT with interventions

1. Optional but recommended: set `config.ee_bounds` in `dk1_base.py`, a box per arm
   (in that arm's base frame, in metres) that keeps the hands above the table.
2. On the GPU server, start the learner:

   ```bash
   bash scripts/dk1_task/run_server.sh
   ```

3. On the robot PC, start the client in one terminal:

   ```bash
   SERVER_HOST=GPU_SERVER_HOSTNAME bash scripts/dk1_task/run_policy.sh
   ```

4. On the robot PC, start the clutch in a second terminal:

   ```bash
   source client_dk1/.venv/bin/activate && python -m client.teleop.clutch
   ```

5. To take over, press **Space** in the clutch terminal and move the leaders. The
   follower moves by the same end-effector motion, starting from where it is, with no
   jump. Its gripper keeps its grip until the leader trigger reaches the same position.
   Press **Space** again to hand back to the policy.
6. Test one takeover early on, in an easy situation, before you rely on it.
7. Label every episode with `1` or `2` in the client terminal. Gradient updates start
   after 10 episodes.

**Check:** the client prints `[clutch] ENGAGED: human control` and `released` when you
press **Space**. In wandb (`expo_ft_dk1`), the success rate climbs over episodes.

## Troubleshooting

| Symptom | What to do |
|---|---|
| `... is not set; run python -m client.dk1_setup_hardware` | `dk1_hardware.json` is missing or incomplete. Rerun the setup script. |
| `reset_joints is not set` | Should not happen with the folded-pose default; check `dk1_base.py`. |
| `DK1 intervention setup failed` at client start | Check the leader ports, and make sure no other process uses UDP port 8765. To run without interventions, set `intervention.enabled = False`. |
| `[intervention] !!! ... RELEASING TO POLICY CONTROL` | The leaders failed to read several times in a row. Check their USB cables and power. |
| One arm moves when the other should | Left and right are swapped. Swap the ports in `dk1_hardware.json`. |
| Camera frames time out | Add `"fourcc": "MJPG"`, lower the resolution, or put the cameras on separate USB controllers. |
| Jerky motion | Make sure `command_interp_hz = 100.0` is set in `dk1_base.py`. |
| Follower lags or feels held back | Raise `max_joint_delta_per_step` (default 0.25); with smoothing on, values below 0.25 felt sluggish. |
| Follower connection fails with the default per-arm classes | The vendored plugin lives in `third_party/trlc-dk1`; patch it there. `use_bimanual_class=True` switches to `BiDK1Follower`, which has a known feature-listing bug that `DK1Env` avoids. |

For design decisions and background, see `plan/001_dk1_port.md`.
