"""Teleoperated demo collection for the bimanual TRLC-DK1 (leader arms -> followers).

Same structure as client/collect_data.py (absl flags, temp-dir routing, only successes are
kept under <save_root>/success/<id>/, streaming MP4 previews), but:

- the controller mirrors the leader joints directly onto the followers (plain joint-space
  teleop, like lerobot-teleoperate), through DK1Env.step so the safety clipping applies;
- the trajectory is written by a small h5py writer in the layout process_droid_dataset and
  scripts/convert_dk1_data_to_lerobot.py read:
      saved_observation/<key>   one per flat observation key (images uint8, prompt bytes)
      action/action14           the *executed* (clipped) 14-D action [L6, Lg, R6, Rg]
      action/leader14           raw leader reading (debug only)
- recording starts on the first leader motion. Do NOT press a bare Enter in this terminal
  while an episode runs: the manual detector reads /dev/tty (1 = success, 2 = reset).

Episode flow: home -> manual scene reset (Enter) -> follower slowly syncs to the leaders ->
move the leaders to start recording -> type 1/2 + Enter to end.

With --pedal_control (default) only the first episode homes. After that, teleop runs
continuously: CLICK the foot pedal = start recording, CLICK again = stop; then open both leader
grippers and squeeze + hold the LEFT one = keep, the RIGHT one = discard (the follower grippers hold
still while you decide; a click instead resumes recording) -> idle (reset the scene, take your time)
-> click to start the next one. The arms never have to go home.

With --nopedal_control --gesture_control only the first episode homes. After that, teleop runs
continuously: come back home (reset_joints, shoulder/elbow joints) and hold still -> recording
stops; squeeze BOTH leader triggers = keep, LEFT alone = discard -> idle (reset the scene, take
your time) -> clearly move an arm -> the next episode starts (with 0.5 s of pre-roll at rest).

    python -m client.collect_data_dk1 --save_root data/dk1_task --num_episodes 20 \
        --task_config configs/task/dk1_task.py
    # no hardware:
    python -m client.collect_data_dk1 --save_root /tmp/dk1_mock --num_episodes 3 \
        --task_config configs/task/dk1_mock.py
"""

import collections
import os
import shutil
import time
from typing import Dict

import cv2
import h5py
import imageio
import numpy as np
from absl import app, flags
from ml_collections import config_flags

from client.envs.dk1_env import LeaderGripperOutcome
from client.teleop.episode_gestures import EpisodeGestureFSM, GestureParams

FLAGS = flags.FLAGS

flags.DEFINE_string("save_root", "data/dk1_task", "Root directory for collected trajectories.")
flags.DEFINE_string("prompt", "", "Task prompt stored with every episode of this session "
                    "(default: the task config's language_instruction).")
flags.DEFINE_integer("num_episodes", 0, "Successful trajectories to collect (0 = run indefinitely).")
config_flags.DEFINE_config_file("task_config", "configs/task/dk1_task.py", "Task config file.", lock_config=False)
flags.DEFINE_bool("test_detector", False, "Never save; on done reset and keep looping (detector testing).")
flags.DEFINE_integer("video_save_width", 320, "Width of saved MP4 frames.")
flags.DEFINE_integer("video_save_height", 240, "Height of saved MP4 frames.")
flags.DEFINE_float("motion_threshold", 0.03, "Leader joint motion (rad) that starts recording.")
flags.DEFINE_float("sync_max_delta", 0.25, "Per-step joint limit (rad) while the follower syncs to the leaders.")
flags.DEFINE_float("sync_tolerance", 0.05, "Follower-leader joint error (rad) that ends the sync phase.")
flags.DEFINE_bool("ee_box", False, "Apply the per-arm EE safety box (hardware ee_bounds) while collecting. Off by "
                  "default: a hold freezes the follower mid-air while the leader moves on. The joint limits "
                  "stay on.")
flags.DEFINE_integer("mock_episode_steps", 30, "mock=True only: steps per scripted episode (auto success).")
flags.DEFINE_bool("pedal_control", True,
                  "Foot-pedal episodes: click = start recording, click = stop, then open both leader grippers "
                  "and squeeze + hold LEFT = keep / RIGHT = discard (task config intervention.leader_outcome). "
                  "No homing between episodes. 1/2 + Enter still work. Overrides --gesture_control.")
flags.DEFINE_bool("gesture_control", True,
                  "Hands-free episodes (client/teleop/episode_gestures.py): recording starts when the "
                  "leaders clearly move and stops when they come back home (reset_joints) and hold "
                  "still; then squeeze BOTH leader triggers = keep, LEFT alone = discard, and it idles "
                  "until you move an arm again. No homing between episodes. 1/2 + Enter still work.")
flags.DEFINE_float("gesture_start_radius", 0.1, "Moving any leader joint this far (rad) from where the arms "
                   "rest starts the next episode.")
flags.DEFINE_float("gesture_arm_radius", 0.6, "Shoulder/elbow (j1-j3) distance (rad) from home the arms must "
                   "reach before an episode can auto-stop.")
flags.DEFINE_float("gesture_stop_radius", 0.4, "Shoulder/elbow (j1-j3) within this (rad) of the home pose "
                   "(task config reset_joints) and still for --gesture_hold_s = episode end.")
flags.DEFINE_float("gesture_hold_s", 0.5, "Stillness at home that ends an episode.")
flags.DEFINE_float("gesture_trigger_s", 0.6, "How long the keep/discard trigger squeeze must be held.")
flags.DEFINE_float("gesture_preroll_s", 0.5, "Seconds of resting frames from before the start kept at the "
                   "beginning of each episode.")


def smallest_missing_id(dir_path: str) -> int:
    os.makedirs(dir_path, exist_ok=True)
    ids = {int(n) for n in os.listdir(dir_path) if n.isdigit() and os.path.isdir(os.path.join(dir_path, n))}
    i = 0
    while i in ids:
        i += 1
    return i


class H5TrajWriter:
    """Append-only, chunked HDF5 writer: one resizable dataset per leaf key."""

    def __init__(self, path: str):
        self.f = h5py.File(path, "w")
        self.n = 0
        self._ds = {}

    def _create(self, name: str, value):
        if isinstance(value, str):
            value = np.bytes_(value.encode("utf-8"))
        arr = np.asarray(value)
        # Strings (the prompt) become fixed-length bytes; process_droid_dataset decodes them.
        dtype = "S256" if arr.dtype.kind in ("U", "S") else arr.dtype
        is_image = arr.dtype == np.uint8 and arr.ndim == 3
        chunk_rows = 1 if is_image else 256
        self._ds[name] = self.f.create_dataset(
            name, shape=(0, *arr.shape), maxshape=(None, *arr.shape), dtype=dtype,
            chunks=(chunk_rows, *arr.shape), compression="lzf" if is_image else None,
        )

    def write_timestep(self, timestep: Dict[str, Dict[str, object]]):
        for group, values in timestep.items():
            for key, value in values.items():
                name = f"{group}/{key}"
                if name not in self._ds:
                    self._create(name, value)
                ds = self._ds[name]
                if isinstance(value, str):
                    value = value.encode("utf-8")[:256]
                ds.resize(self.n + 1, axis=0)
                ds[self.n] = value
        self.n += 1

    def close(self, metadata: dict | None = None):
        if metadata:
            for k, v in metadata.items():
                try:
                    self.f.attrs[k] = v
                except TypeError:
                    self.f.attrs[k] = str(v)
        self.f.close()


class LeaderMirror:
    """Joint-space teleop: follower target = leader joints (through the configured joint map)."""

    def __init__(self, env, task_config, mock_steps: int = 30):
        self.env = env
        self.leaders = env.get_leaders()
        iv = task_config.get("intervention") or {}
        off = iv.get("leader_joint_offsets")
        sgn = iv.get("leader_joint_signs")
        self.offsets = np.zeros(12) if off is None else np.asarray(off, float).reshape(12)
        self.signs = np.ones(12) if sgn is None else np.asarray(sgn, float).reshape(12)
        self.mock = bool(getattr(env, "mock", False))
        self.mock_steps = int(mock_steps)
        self._t = 0
        self._mock_home = None

    def reset_state(self):
        self._t = 0
        if self.mock:
            self._mock_home = self.env.get_state().copy()
            self.leaders.pose = self._mock_home.copy()

    def _scripted_leader(self):
        """mock: a smooth bimanual motion around the start pose, grippers close half-way through."""
        t = self._t / max(self.env.control_hz, 1.0)
        pose = self._mock_home.copy()
        wiggle = 0.25 * np.sin(2 * np.pi * 0.3 * t)
        pose[0:6] += wiggle * np.array([1.0, 0.4, -0.4, 0.5, 0.5, 0.8])
        pose[7:13] += wiggle * np.array([-1.0, 0.4, -0.4, -0.5, 0.5, -0.8])
        closed = 1.0 if self._t > self.mock_steps // 2 else 0.0
        pose[6] = pose[13] = closed
        self.leaders.pose = pose

    def get_leader(self) -> np.ndarray:
        if self.mock:
            self._scripted_leader()
        return self.leaders.get()

    def to_follower(self, leader14: np.ndarray) -> np.ndarray:
        out = np.asarray(leader14, float).copy()
        out[0:6] = self.signs[0:6] * out[0:6] + self.offsets[0:6]
        out[7:13] = self.signs[6:12] * out[7:13] + self.offsets[6:12]
        return out

    def forward(self):
        leader = self.get_leader()
        self._t += 1
        return self.to_follower(leader), leader


def _sync_follower_to_leaders(env, controller, max_delta: float, tol: float, timeout_s: float = 20.0):
    """Slowly bring the followers onto the leader pose before recording (avoids a jump)."""
    dt = 1.0 / env.control_hz
    t_end = time.time() + timeout_s
    arm_idx = np.r_[0:6, 7:13]
    printed = 0.0
    while time.time() < t_end:
        t0 = time.time()
        target = controller.to_follower(controller.leaders.get())
        err = np.max(np.abs(target[arm_idx] - env.get_state()[arm_idx]))
        if err < tol:
            return True
        env.step(target, max_delta=max_delta)
        if time.time() - printed > 1.0:
            print(f"[sync] follower -> leaders, max joint error {err:.3f} rad")
            printed = time.time()
        time.sleep(max(0.0, dt - (time.time() - t0)))
    print("[sync] timeout; continuing (per-step clipping still applies).")
    return False


def _joint_name(i12: int) -> str:
    return f"{'L' if i12 < 6 else 'R'}j{i12 % 6 + 1}"


def _gesture_status(fsm, rec_s: float) -> str:
    s, p, last = fsm.state, fsm.p, fsm.last
    g = last.get("grips", np.zeros(2))
    trig = f"triggers {g[0]:.2f}/{g[1]:.2f}"
    j = last["far_joint"]
    home = (f"from home {last['dist']:.2f} {'ok' if last['dist'] < p.stop_radius else 'X'} (< {p.stop_radius:.2f}; "
            f"farthest {_joint_name(j)} now {last['q'][j]:+.2f} home {last['home'][j]:+.2f})")
    speed = (f"speed {last['speed']:.2f} {'ok' if last['speed'] < p.still_speed else 'X'} "
             f"(< {p.still_speed:.2f} rad/s; fastest {_joint_name(last['fast_joint'])})")
    if s == "idle" and not fsm.ready:
        return f"[gesture IDLE] waiting for the arms at home, still, triggers open | {home} | {speed} | {trig}"
    if s == "idle":
        return f"[gesture IDLE] ready: move an arm to start | moved {last['moved']:.2f} (start > {p.start_radius:.2f})"
    if s == "decide":
        return f"[gesture DECIDE] squeeze BOTH = keep, LEFT alone = discard | {trig}"
    return (f"[gesture REC {rec_s:5.1f}s] armed={'yes' if fsm.armed else f'no (go > {p.arm_radius:.2f} from home)'} | "
            f"{home} | {speed} | still {fsm._count / fsm.hz:.1f}/{p.hold_s:.1f}s")


_PEDAL_HELP = ("CLICK again to stop; then open both leader grippers and squeeze + hold LEFT = keep, "
               "RIGHT = discard. (1/2 + Enter still work.)")


def _leader_outcome_kwargs(task_config) -> dict:
    iv = task_config.get("intervention", None) or {}
    kw = dict(iv.get("leader_outcome", None) or {})
    kw.pop("enabled", None)
    return kw


def _open_pedal(env, task_config):
    """The foot pedal as the episode start/stop clicker (env._pedal too, so a click also answers the
    "reset the scene" prompt). The env's own leader-gripper outcome is turned off here: while
    collecting, the leader grippers are for grasping."""
    from client.teleop.pedal import DEFAULT_PEDAL_NAME, Pedal

    iv = task_config.get("intervention", None) or {}
    pedal = Pedal(iv.get("pedal_name", DEFAULT_PEDAL_NAME), clicker=True)
    if getattr(env, "_pedal", None) is None:
        env._pedal = pedal
    env._leader_outcome = None
    return pedal


def collect_trajectory(env, controller, save_filepath=None, recording_folderpath=None, test_detector=False):
    task_config = FLAGS.task_config

    pedal_mode = FLAGS.pedal_control and not controller.mock and not test_detector
    gesture = FLAGS.gesture_control and not pedal_mode and not controller.mock and not test_detector
    fsm = getattr(controller, "gesture_fsm", None)
    pedal = getattr(controller, "pedal", None)
    if pedal_mode and pedal is None:
        # Before env.reset(): a pedal click then also answers the first "reset the scene" prompt.
        pedal = controller.pedal = _open_pedal(env, task_config)
    if (pedal_mode and not getattr(controller, "pedal_started", False)) or (
            not pedal_mode and (not gesture or fsm is None)):
        # Home + manual scene reset + sync. In gesture mode only before the first episode; after
        # that teleop runs continuously and the previous episode's end is the next start.
        t_reset0 = time.perf_counter()
        env.reset()
        controller.reset_state()
        print("[between-episode] env.reset()={:.2f}s".format(time.perf_counter() - t_reset0))
        if not controller.mock:
            _sync_follower_to_leaders(env, controller, FLAGS.sync_max_delta, FLAGS.sync_tolerance)
        if pedal_mode:
            controller.pedal_started = True
        if gesture:
            fsm = controller.gesture_fsm = EpisodeGestureFSM(env.control_hz, GestureParams(
                start_radius=FLAGS.gesture_start_radius, arm_radius=FLAGS.gesture_arm_radius,
                stop_radius=FLAGS.gesture_stop_radius, hold_s=FLAGS.gesture_hold_s,
                gesture_s=FLAGS.gesture_trigger_s),
                home=np.asarray(env.reset_joints, dtype=np.float64)[np.r_[0:6, 7:13]])

    traj_writer = H5TrajWriter(save_filepath) if save_filepath else None
    mp4_writers = {}
    start_recording = controller.mock  # mock: record from the first step
    leader_start = controller.to_follower(controller.leaders.get())
    episode_success = None
    end_reason = "manual"
    last_control_start = None
    loop_ms = []
    n_steps = 0
    stopped = False  # gesture mode: back home, recording over, waiting for keep/discard
    preroll = collections.deque(maxlen=max(1, int(round(FLAGS.gesture_preroll_s * env.control_hz))))
    pedal_state = "idle"  # pedal mode: idle -> record -> decide -> (keep/discard) idle
    clicks_seen = pedal.clicks() if pedal_mode else 0
    outcome = LeaderGripperOutcome(**_leader_outcome_kwargs(task_config)) if pedal_mode else None
    frozen_grips = None  # pedal mode, decide: follower gripper commands held while the leaders squeeze

    def write_frame(saved_obs, executed, leader):
        nonlocal n_steps
        if recording_folderpath and not mp4_writers:
            video_dir = os.path.join(os.path.dirname(recording_folderpath), "recordings", "MP4")
            os.makedirs(video_dir, exist_ok=True)
            for key, val in saved_obs.items():
                if isinstance(val, np.ndarray) and val.ndim == 3 and val.shape[-1] == 3:
                    mp4_writers[key] = imageio.get_writer(
                        os.path.join(video_dir, f"{key}.mp4"), fps=int(env.control_hz), format="ffmpeg",
                        codec="libx264", output_params=["-preset", "ultrafast", "-crf", "20"],
                    )
        n_steps += 1
        if traj_writer is not None:
            traj_writer.write_timestep({
                "saved_observation": saved_obs,
                "action": {
                    "action14": np.asarray(executed, dtype=np.float32),
                    "leader14": np.asarray(leader, dtype=np.float32),
                },
            })
        vid_size = (FLAGS.video_save_width, FLAGS.video_save_height)
        for key, writer in mp4_writers.items():
            f = np.asarray(saved_obs[key], dtype=np.uint8)
            if (f.shape[1], f.shape[0]) != vid_size:
                f = cv2.resize(f, vid_size, interpolation=cv2.INTER_LINEAR)
            writer.append_data(f)

    if pedal_mode:
        print(f"[pedal] Idle. Reset the scene, then CLICK the pedal to start recording. {_PEDAL_HELP}")
    elif gesture:
        print("[gesture] Idle. Reset the scene, then move an arm to start recording. Come back home and "
              "hold still to stop, then squeeze BOTH triggers = keep, LEFT alone = discard. (1/2 + Enter still work.)")
    else:
        print("Move the leader arms to start recording. Type 1 + Enter = success, 2 + Enter = discard.")
    try:
        while True:
            t_start = time.perf_counter()
            raw = env.get_raw_observation()
            t_cam = time.perf_counter()
            saved_obs = env.transform_observation(raw)
            t_tf = time.perf_counter()
            done, success, _, _ = env.get_info_for_step(raw)
            if controller.mock and n_steps >= FLAGS.mock_episode_steps:
                done, success = True, True
            t_info = time.perf_counter()

            if done:
                if test_detector:
                    print(f"Done (success={success}); test_detector=True")
                    env.reset()
                    controller.reset_state()
                    continue
                episode_success = success
                if gesture:
                    fsm.reset()  # keyboard end: back to idle
                return {"success": success, "failure": not success, "steps": n_steps}

            action, leader = controller.forward()
            if frozen_grips is not None:
                action = np.asarray(action, dtype=np.float64).copy()
                action[[6, 13]] = frozen_grips
            t_ctrl = time.perf_counter()

            sleep_left = 0.0 if last_control_start is None else (1.0 / env.control_hz) - (time.perf_counter() - last_control_start)
            if sleep_left > 0:
                time.sleep(sleep_left)
            t_sleep = time.perf_counter()
            last_control_start = t_sleep
            info = env.step(action)  # same per-step cap as RL rollouts
            t_step = time.perf_counter()

            event = None
            if pedal_mode:
                clicks = pedal.clicks()
                clicked = clicks != clicks_seen
                clicks_seen = clicks
                if pedal_state == "idle" and clicked:
                    pedal_state, start_recording = "record", True
                    event = "start"
                elif pedal_state == "record" and clicked:
                    pedal_state, event = "decide", "stop"
                    frozen_grips = np.asarray(info["executed_action"], dtype=np.float64)[[6, 13]].copy()
                    outcome.disarm()
                elif pedal_state == "decide" and clicked:
                    pedal_state, stopped, frozen_grips = "record", False, None
                    print("[pedal] Resumed recording. CLICK to stop.")
                elif pedal_state == "decide":
                    result = outcome.update(leader)
                    if result is not None:
                        event = "keep" if result == "success" else "discard"
            elif gesture:
                event = fsm.update(controller.to_follower(leader)[np.r_[0:6, 7:13]], leader[[6, 13]])
                if event == "ready":
                    print("[gesture] Ready: move an arm to start recording.")
                elif event == "start":
                    start_recording = True
                    for frame in preroll:  # the resting frames just before the motion
                        write_frame(*frame)
                    preroll.clear()
                elif fsm.state == "idle" and fsm.ready:
                    preroll.append((saved_obs, info["executed_action"], leader))
            elif not start_recording:
                moved = np.max(np.abs(controller.to_follower(leader) - leader_start)[np.r_[0:6, 7:13]])
                grip = np.max(np.abs(leader[[6, 13]] - leader_start[[6, 13]]))
                if moved > FLAGS.motion_threshold or grip > 0.1:
                    start_recording = True
                    print("start recording")
            if event == "start":
                print("start recording")

            if start_recording and not stopped:
                write_frame(saved_obs, info["executed_action"], leader)
            t_rec = time.perf_counter()

            if event == "stop" and pedal_mode:
                stopped = True
                print(f"[pedal] Recording stopped after {n_steps / env.control_hz:.1f} s. Open both leader "
                      f"grippers, then squeeze + hold the {outcome.success_side.upper()} one = KEEP, the other = "
                      "DISCARD (CLICK = resume recording).")
            elif event == "stop":
                stopped = True
                print(f"[gesture] Back home after {n_steps / env.control_hz:.1f} s; recording stopped. "
                      "Squeeze BOTH triggers = keep, LEFT alone = discard.")
            elif event in ("keep", "discard"):
                episode_success = event == "keep"
                end_reason = event
                tag = "[pedal]" if pedal_mode else "[gesture]"
                nxt = "CLICK the pedal" if pedal_mode else "move an arm"
                print(f"{tag} {event.upper()}. Idle: reset the scene, then {nxt} to start the next episode.")
                return {"success": episode_success, "failure": not episode_success, "steps": n_steps,
                        "end_reason": end_reason}

            ms = lambda a, b: (b - a) * 1000.0  # noqa: E731
            work = ms(t_start, t_rec) - ms(t_ctrl, t_sleep)
            loop_ms.append(work)
            if pedal_mode:
                if pedal_state == "decide" and len(loop_ms) % max(1, int(env.control_hz * 2.0)) == 0:
                    print(f"[pedal] deciding: leader grippers L={leader[6]:.2f} R={leader[13]:.2f} "
                          f"({'armed' if outcome._armed else f'open both below {outcome.open_thresh:.2f} first'}; "
                          f"squeeze > {outcome.close_thresh:.2f} for {outcome.hold_s:.1f} s)", flush=True)
                continue
            if gesture:
                # A 2 Hz status line instead of per-step timing, so the reason a stop is not
                # firing is visible (loop-rate summary still prints at the end of the episode).
                every = 5.0 if fsm.state == "idle" and fsm.ready else 0.5  # quiet while you reset the scene
                if len(loop_ms) % max(1, int(env.control_hz * every)) == 0:
                    print(_gesture_status(fsm, n_steps / env.control_hz), flush=True)
                continue
            print(
                f"timing ms: read_obs={ms(t_start, t_cam):.1f} transform={ms(t_cam, t_tf):.1f} "
                f"info={ms(t_tf, t_info):.1f} leader={ms(t_info, t_ctrl):.1f} sleep={ms(t_ctrl, t_sleep):.1f} "
                f"env_step={ms(t_sleep, t_step):.1f} record={ms(t_step, t_rec):.1f} | work={work:.1f} "
                f"(budget {1000.0 / env.control_hz:.1f})"
            )
    finally:
        for w in mp4_writers.values():
            try:
                w.close()
            except Exception:
                pass
        if traj_writer is not None:
            traj_writer.close(metadata={"language_instruction": task_config.language_instruction,
                                        "control_hz": env.control_hz, "success": bool(episode_success),
                                        "end_reason": end_reason})
        if loop_ms:
            arr = np.asarray(loop_ms)
            budget = 1000.0 / env.control_hz
            print(f"[loop rate] work mean {arr.mean():.1f} ms, p95 {np.percentile(arr, 95):.1f} ms, "
                  f"max {arr.max():.1f} ms; budget {budget:.1f} ms; over budget {np.mean(arr > budget) * 100:.0f}% of steps")


def run_and_route_one(env, controller, base_dir) -> Dict[str, object]:
    tmp_dir = os.path.join(base_dir, "tmp", f"session_{int(time.time() * 1000)}")
    images_dir = os.path.join(tmp_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    save_filepath = os.path.join(tmp_dir, "traj.hdf5")
    print("Temp traj file:", save_filepath)

    result = collect_trajectory(env, controller, save_filepath=save_filepath, recording_folderpath=images_dir,
                                test_detector=FLAGS.test_detector)
    success = bool(result.get("success")) and result.get("steps", 0) > 0
    print(f"Outcome: {'success' if success else 'failure'} ({result.get('steps', 0)} recorded steps)")
    if not success:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("Discarded temp data.")
        return {"dest_dir": None, "id": None, "result": result}

    outcome_root = os.path.join(base_dir, "success")
    new_id = smallest_missing_id(outcome_root)
    dest_dir = os.path.join(outcome_root, str(new_id))
    shutil.move(tmp_dir, dest_dir)
    print("Result saved:", os.path.join(dest_dir, "traj.hdf5"))
    return {"dest_dir": dest_dir, "id": new_id, "result": result}


def main(_):
    base_dir = FLAGS.save_root
    os.makedirs(base_dir, exist_ok=True)
    task_config = FLAGS.task_config
    if FLAGS.prompt:
        task_config.language_instruction = FLAGS.prompt
    print(f"Prompt for this session: {task_config.language_instruction!r}")
    env_kwargs = dict(task_config)
    env_kwargs["video_dir"] = ""
    env_kwargs["env_usage"] = "collect"
    if not FLAGS.ee_box:
        env_kwargs["ee_bounds"] = None
    env = task_config.env(**env_kwargs)
    env.ignore_auto_reset = True  # teleop episodes end on the manual detector only
    controller = LeaderMirror(env, task_config, mock_steps=FLAGS.mock_episode_steps)

    try:
        if FLAGS.test_detector:
            collect_trajectory(env, controller, test_detector=True)
            return
        episode = successful = 0
        while True:
            episode += 1
            print(f"\n{'=' * 60}\nEpisode #{episode} (successful {successful}/"
                  f"{FLAGS.num_episodes if FLAGS.num_episodes > 0 else 'inf'})\n{'=' * 60}\n")
            result = run_and_route_one(env, controller, base_dir)
            if result["dest_dir"] is not None:
                successful += 1
            if FLAGS.num_episodes > 0 and successful >= FLAGS.num_episodes:
                print(f"Reached {FLAGS.num_episodes} successful episodes after {episode} attempts")
                break
    finally:
        env.close()


if __name__ == "__main__":
    app.run(main)
