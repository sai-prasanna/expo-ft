"""Convert DK1 demos (client/collect_data_dk1.py traj.hdf5 files) to a LeRobot dataset.

Runs in the SERVER venv (openpi's pinned LeRobot, `lerobot.common.*` API, v2.x format).

Feature names equal the env observation keys (client/envs/dk1_env.py), which the openpi
`LeRobotDK1DataConfig` repack maps to themselves:
    head_image, left_wrist_image, right_wrist_image : image (H, W, 3), H, W = task_config.image_size
    state   : float32 (14,)
    actions : float32 (14,)   = action/<task_config.action_space> (the executed 14-D action)
fps = task_config.control_hz. The task (prompt) of each episode is the prompt stored in its
traj.hdf5 (collect_data_dk1.py --prompt), falling back to task_config.language_instruction.

    uv run scripts/convert_dk1_data_to_lerobot.py --data_dir data/dk1_task/success \
        --repo_name expo_ft/dk1_task_20 --task_config configs/task/dk1_task.py --max_episodes 20
"""

import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from tqdm import tqdm
import tyro

from convert_droid_data_to_lerobot import load_task_config  # sibling script (scripts/ is sys.path[0])

IMAGE_KEYS = ("head_image", "left_wrist_image", "right_wrist_image")
STATE_KEY = "state"


def _resize(image: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    if image.shape[:2] == tuple(hw):
        return image
    return np.array(Image.fromarray(image).resize((hw[1], hw[0]), resample=Image.BICUBIC))


def _has_trajectory(p: Path) -> bool:
    try:
        with h5py.File(p, "r") as f:
            return "saved_observation" in f and "action" in f and len(f["action"]) > 0
    except OSError:
        return False


def main(
    data_dir: str,
    *,
    repo_name: str,
    task_config: str,
    max_episodes: int | None = None,
    push_to_hub: bool = False,
):
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    cfg = load_task_config(task_config)
    action_key = cfg.action_space
    if cfg.get("gripper_action_space") is not None:
        raise ValueError("DK1 conversion expects gripper_action_space=None (gripper inside action14).")
    hw = tuple(cfg.image_size)
    action_dim = 14

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="trlc_dk1_bimanual",
        fps=int(cfg.control_hz),
        features={
            **{k: {"dtype": "image", "shape": (*hw, 3), "names": ["height", "width", "channel"]} for k in IMAGE_KEYS},
            STATE_KEY: {"dtype": "float32", "shape": (14,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (action_dim,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    def _sort_key(p):
        name = p.parent.name
        return (int(name),) if name.isdigit() else (float("inf"), name)

    paths = sorted(Path(data_dir).glob("**/traj.hdf5"), key=_sort_key)
    paths = [p for p in paths if _has_trajectory(p)]
    if max_episodes is not None:
        paths = paths[:max_episodes]
    print(f"Converting {len(paths)} episodes; fps={cfg.control_hz}")
    prompt_counts = {}

    for path in tqdm(paths, desc="Converting episodes"):
        with h5py.File(path, "r") as f:
            obs = {k: f["saved_observation"][k][:] for k in (*IMAGE_KEYS, STATE_KEY)}
            actions = np.asarray(f["action"][action_key][:], dtype=np.float32)
            prompt = cfg.language_instruction
            if "prompt" in f["saved_observation"] and len(f["saved_observation"]["prompt"]) > 0:
                stored = f["saved_observation"]["prompt"][0]
                prompt = stored.decode("utf-8") if isinstance(stored, bytes) else str(stored)
        prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1
        if actions.ndim != 2 or actions.shape[1] != action_dim:
            raise ValueError(f"{path}: action/{action_key} has shape {actions.shape}, expected (T, 14)")
        for t in range(len(actions)):
            frame = {k: _resize(np.asarray(obs[k][t], dtype=np.uint8), hw) for k in IMAGE_KEYS}
            frame[STATE_KEY] = np.asarray(obs[STATE_KEY][t], dtype=np.float32)
            frame["actions"] = actions[t]
            frame["task"] = prompt
            dataset.add_frame(frame)
        dataset.save_episode()

    for p, n in prompt_counts.items():
        print(f"  {n:4d} episodes: {p!r}")
    print(f"Saved LeRobot dataset to {output_path}")
    if push_to_hub:
        dataset.push_to_hub(tags=["trlc_dk1", "bimanual"], private=True, push_videos=True)


if __name__ == "__main__":
    tyro.cli(main)
