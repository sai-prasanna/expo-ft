import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_droid_example(*, use_cartesian_state: bool = False) -> dict:
    """Creates a random input example for the Droid policy.

    use_cartesian_state: if True, include observation/cartesian_position (6 dims: x,y,z + euler).
    Else observation/joint_position (7 joint dims). Gripper is always observation/gripper_position.
    """
    state_key = "observation/cartesian_position" if use_cartesian_state else "observation/joint_position"
    pose_dim = 6 if use_cartesian_state else 7  # cartesian: 6 (3 pos + 3 euler); joint: 7
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        state_key: np.random.rand(pose_dim),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType
    # Feed observation/exterior_image_2_left to the third slot instead of masked zeros.
    use_second_exterior: bool = False

    def __call__(self, data: dict) -> dict:
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            gripper_pos = gripper_pos[np.newaxis]
        # Support either joint (7d) or cartesian (e.g. 6d pose) as state; cartesian is used when present.
        if "observation/cartesian_position" in data:
            pose = np.asarray(data["observation/cartesian_position"]).flatten()
            state = np.concatenate([pose, gripper_pos])
        else:
            state = np.concatenate([data["observation/joint_position"], gripper_pos])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])
        third_image = (
            _parse_image(data["observation/exterior_image_2_left"])
            if self.use_second_exterior else None
        )

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                if third_image is None:
                    images = (base_image, wrist_image, np.zeros_like(base_image))
                    image_masks = (np.True_, np.True_, np.False_)
                else:
                    images = (base_image, wrist_image, third_image)
                    image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    """Output transform that returns the first `action_dim` action dimensions.

    Use action_dim=8 for joint velocity (7 joint + 1 gripper), action_dim=7 for
    cartesian velocity (6D + 1 gripper).
    """

    action_dim: int = 8

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
