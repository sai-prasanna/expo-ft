"""Detect DK1 serial ports and cameras and write configs/task/dk1_hardware.json.

Run on the robot machine with the client_dk1 venv, from the repo root:

    python -m client.dk1_setup_hardware                 # ports + cameras (interactive)
    python -m client.dk1_setup_hardware --only ports    # just the 4 serial ports
    python -m client.dk1_setup_hardware --only cameras  # just the 3 cameras
    python -m client.dk1_setup_hardware --record_home   # teleop to the home pose, save reset_joints
    python -m client.dk1_setup_hardware --check         # open everything listed in the file

Serial ports: unplug-and-replug identification (like `lerobot-find-port`), which also settles
left vs right. Devices are stored by their stable /dev/serial/by-id path (or by-path when two
adapters share an id), so the file survives reboots and replugging into the same USB port.

Cameras: every capture device is opened, a frame is saved to <out_dir>/camera_<n>.jpg (and
shown if a display is available), and you assign head / left_wrist / right_wrist (or skip).
Stored by /dev/v4l/by-id (or by-path) as well.

The file is meant to be committed; edit it by hand to correct anything (e.g. swap left/right,
change resolution/fps, add "rotation": 180 or "fourcc": "MJPG" to a camera).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

HARDWARE_FILE = Path(__file__).resolve().parents[1] / "configs" / "task" / "dk1_hardware.json"
CAMERA_ROLES = ("head", "left_wrist", "right_wrist")
PORT_ROLES = ("left_follower_port", "right_follower_port", "left_leader_port", "right_leader_port")
PORT_HELP = {
    "left_follower_port": "LEFT FOLLOWER arm (DAMIAO / CAN-over-serial)",
    "right_follower_port": "RIGHT FOLLOWER arm (DAMIAO / CAN-over-serial)",
    "left_leader_port": "LEFT LEADER arm (Dynamixel)",
    "right_leader_port": "RIGHT LEADER arm (Dynamixel)",
}


def load() -> dict:
    if HARDWARE_FILE.exists():
        return json.loads(HARDWARE_FILE.read_text())
    return {}


def save(hw: dict):
    HARDWARE_FILE.write_text(json.dumps(hw, indent=2) + "\n")
    print(f"\nWrote {HARDWARE_FILE}")


# --------------------------------------------------------------------------- stable names
def _stable_name(dev: str, dirs: list[str], n_devices: int, pattern: str = "*") -> str:
    """Stable symlink for `dev`: by-id if every device has its own by-id link, else by-path.

    Identical adapters without serial numbers share one by-id name (udev keeps only the last),
    so if there are fewer by-id links than devices, by-id is ambiguous and by-path (physical
    USB port) is used instead.
    """
    real = os.path.realpath(dev)
    for d in dirs:
        links = glob.glob(os.path.join(d, pattern))
        if "by-id" in d and len({os.path.realpath(p) for p in links}) < n_devices:
            continue
        mine = [p for p in links if os.path.realpath(p) == real]
        if mine:
            return sorted(mine)[0]
    return dev


def _serial_ports() -> set[str]:
    return set(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))


def _input(msg: str) -> str:
    try:
        return input(msg)
    except EOFError:
        return ""


def setup_ports(hw: dict):
    print("\n=== Serial ports ===")
    print("All four DK1 USB cables (2 followers, 2 leaders) should be plugged in and powered.")
    before = _serial_ports()
    print("Detected:", sorted(before) or "none")
    for role in PORT_ROLES:
        while True:
            _input(f"\nUnplug the USB cable of the {PORT_HELP[role]} and press Enter (Enter again to skip)... ")
            time.sleep(0.5)
            now = _serial_ports()
            gone = sorted(before - now)
            if len(gone) == 1:
                dev = gone[0]
                _input(f"  -> {dev}. Plug it back in and press Enter... ")
                for _ in range(20):
                    time.sleep(0.25)
                    if dev in _serial_ports() or len(_serial_ports()) == len(before):
                        break
                after = _serial_ports()
                new = sorted(after - now)
                dev = new[0] if len(new) == 1 else dev  # may re-enumerate under a new name
                stable = _stable_name(dev, ["/dev/serial/by-id", "/dev/serial/by-path"], len(after))
                hw[role] = stable
                print(f"  {role} = {stable}")
                before = after
                break
            if not gone:
                ans = _input("  No port disappeared. Retry? [Y/n] ").strip().lower()
                if ans == "n":
                    print(f"  skipped {role}")
                    break
            else:
                print(f"  Several ports disappeared ({gone}); plug everything back in and retry.")
                _input("  Press Enter when all cables are back... ")
                before = _serial_ports()


def _camera_devices() -> list[str]:
    devs = []
    by_id = sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
    by_path = sorted(glob.glob("/dev/v4l/by-path/*-video-index0"))
    seen = set()
    for p in by_id + by_path:
        real = os.path.realpath(p)
        if real not in seen:
            seen.add(real)
            devs.append(real)
    if not devs:  # no udev symlinks: fall back to all /dev/video*
        devs = sorted(glob.glob("/dev/video*"), key=lambda s: int(s.replace("/dev/video", "") or 0))
    return devs


def setup_cameras(hw: dict, out_dir: Path, width: int, height: int, fps: int):
    import cv2

    print("\n=== Cameras ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    can_show = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    found = []
    for dev in _camera_devices():
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = False, None
        for _ in range(10):
            ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            continue
        n = len(found)
        path = out_dir / f"camera_{n}.jpg"
        cv2.imwrite(str(path), frame)
        found.append((dev, path, frame))
        print(f"  camera_{n}: {dev}  ({frame.shape[1]}x{frame.shape[0]})  preview: {path}")
    if not found:
        print("No working cameras found.")
        return

    cams = dict(hw.get("cameras", {}))
    for n, (dev, path, frame) in enumerate(found):
        if can_show:
            try:
                cv2.imshow(f"camera_{n}", frame)
                cv2.waitKey(500)
            except cv2.error:
                can_show = False
        role = ""
        while role not in CAMERA_ROLES and role != "skip":
            role = _input(f"Role of camera_{n} ({dev}) [{'/'.join(CAMERA_ROLES)}/skip]: ").strip() or "skip"
        if can_show:
            cv2.destroyAllWindows()
        if role == "skip":
            continue
        stable = _stable_name(dev, ["/dev/v4l/by-id", "/dev/v4l/by-path"], len(found), "*-video-index0")
        cams[role] = {"index_or_path": stable, "width": width, "height": height, "fps": fps}
        print(f"  {role} = {stable}")
    missing = [r for r in CAMERA_ROLES if r not in cams]
    if missing:
        print(f"WARNING: no camera assigned to {missing}")
    hw["cameras"] = cams


def check(hw: dict):
    from client.envs.dk1_env import _RealArms, _RealCameras, _RealLeaders

    print("Opening cameras...")
    cams = _RealCameras(hw["cameras"])
    for name, img in cams.read().items():
        print(f"  {name}: {img.shape} {img.dtype}")
    cams.close()
    print("Connecting followers...")
    arms = _RealArms(hw["left_follower_port"], hw["right_follower_port"])
    print("  follower state:", arms.get().round(3))
    input("Motors will be DISABLED next and the arms will fall: support both arms, then press Enter.")
    arms.close()
    print("Connecting leaders...")
    leaders = _RealLeaders(hw["left_leader_port"], hw["right_leader_port"])
    print("  leader state:", leaders.get().round(3))
    leaders.close()
    print("OK")


def record_home(hw: dict, hz: float = 50.0):
    """Mirror leaders -> followers; on Enter save the follower pose as reset_joints."""
    import select
    import sys

    import numpy as np

    from client.envs.dk1_env import _RealArms, _RealLeaders, clip_joint_delta, clip_to_limits
    from configs.task import dk1_base  # leader->follower joint map lives in the task config
    iv = dict(dk1_base.get_config().intervention)
    offsets = np.asarray(iv.get("leader_joint_offsets", [0.0] * 12), float)
    signs = np.asarray(iv.get("leader_joint_signs", [1.0] * 12), float)

    def to_follower(l14):
        out = l14.copy()
        out[0:6] = signs[0:6] * out[0:6] + offsets[0:6]
        out[7:13] = signs[6:12] * out[7:13] + offsets[6:12]
        return out

    leaders = _RealLeaders(hw["left_leader_port"], hw["right_leader_port"])
    arms = _RealArms(hw["left_follower_port"], hw["right_follower_port"])
    print("Teleop running (followers ease onto the leaders). Move to the home pose and press Enter.")
    try:
        while True:
            t0 = time.time()
            target = clip_to_limits(to_follower(leaders.get()))
            arms.send(clip_joint_delta(target, arms.get(), 0.02))
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                break
            time.sleep(max(0.0, 1.0 / hz - (time.time() - t0)))
        home = arms.get()
        home[6] = home[13] = 0.0
        hw["reset_joints"] = [round(float(x), 4) for x in home]
        print("reset_joints =", hw["reset_joints"])
        input("Motors will be DISABLED next and the arms will fall: support both arms, then press Enter.")
    finally:
        leaders.close()
        arms.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=["ports", "cameras"])
    p.add_argument("--record_home", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--preview_dir", default="outputs/dk1_camera_previews")
    args = p.parse_args()

    hw = load()
    if args.check:
        check(hw)
        return
    if args.record_home:
        record_home(hw)
        save(hw)
        return
    if args.only in (None, "ports"):
        setup_ports(hw)
    if args.only in (None, "cameras"):
        setup_cameras(hw, Path(args.preview_dir), args.width, args.height, args.fps)
    save(hw)
    print(json.dumps(hw, indent=2))


if __name__ == "__main__":
    main()
