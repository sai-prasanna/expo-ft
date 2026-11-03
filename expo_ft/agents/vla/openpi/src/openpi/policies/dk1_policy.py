"""Policy transforms for the bimanual TRLC-DK1.

Layout (state and actions): 14-D absolute joint positions
    [L joint_1..joint_6, L gripper, R joint_1..joint_6, R gripper]
Joints are radians; grippers run 0 (open) .. 1 (closed). This is the DK1 LeRobot
plugin's own convention, so unlike `aloha_policy` there are no joint flips or
gripper conversions (`adapt_to_pi`), and no delta actions.

Input keys are flat and equal to the env observation keys / LeRobot feature names
(the repack transform maps them to themselves):
    head_image, left_wrist_image, right_wrist_image, state, [actions], [prompt]
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

DK1_ACTION_DIM = 14
DK1_IMAGE_KEYS = ("head_image", "left_wrist_image", "right_wrist_image")


def make_dk1_example() -> dict:
    """Creates a random input example for the DK1 policy."""
    return {
        "head_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "state": np.random.rand(DK1_ACTION_DIM).astype(np.float32),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    # LeRobot stores images as float32 (C,H,W) in [0, 1]; the env sends uint8 (H,W,C).
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DK1Inputs(transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        head = _parse_image(data["head_image"])
        left = _parse_image(data["left_wrist_image"])
        right = _parse_image(data["right_wrist_image"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            case _:
                raise ValueError(f"Unsupported model type for DK1: {self.model_type}")

        # State stays 14-D here; PadStatesAndActions (model transforms) pads to model.action_dim
        # after normalization, so norm stats are 14-D.
        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": dict(zip(names, (head, left, right), strict=True)),
            "image_mask": dict(zip(names, (np.True_, np.True_, np.True_), strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class DK1Outputs(transforms.DataTransformFn):
    action_dim: int = DK1_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
