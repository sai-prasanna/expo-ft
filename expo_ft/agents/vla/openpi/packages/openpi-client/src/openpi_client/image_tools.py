# Uses cv2 instead of PIL for faster resizing 
import numpy as np
import cv2


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def _resize_with_pad_single(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """One image (H, W, C): same logic as PIL version — ratio, resized size, center pad with 0.
    Output shape (height, width, C), dtype uint8, range [0, 255]. Pixel values may differ slightly from PIL due to interpolation."""
    cur_h, cur_w = img.shape[0], img.shape[1]
    if cur_h == height and cur_w == width:
        return np.asarray(img, dtype=np.uint8)
    # Same formula as PIL: ratio = max(cur_width/width, cur_height/height)
    ratio = max(cur_w / width, cur_h / height)
    resized_w = int(cur_w / ratio)
    resized_h = int(cur_h / ratio)
    resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    if resized.dtype != np.uint8:
        resized = np.clip(np.round(resized), 0, 255).astype(np.uint8)
    # Same center pad as PIL: pad_height = int((height - resized_height) / 2), pad_width = int((width - resized_width) / 2)
    pad_w = int((width - resized_w) / 2)
    pad_h = int((height - resized_h) / 2)
    out = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
    out[pad_h : pad_h + resized_h, pad_w : pad_w + resized_w] = resized
    return out


def resize_with_pad(images: np.ndarray, height: int, width: int, method=None) -> np.ndarray:
    """Replicates tf.image.resize_with_pad. Resizes to target (height, width), aspect ratio preserved, center-pad with 0.

    Output shape, dtype (uint8), value range [0, 255], and pad/ratio logic match the original PIL implementation.
    Pixel values may differ slightly from PIL due to cv2 interpolation.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: Unused (kept for API compatibility).

    Returns:
        The resized images in [..., height, width, channel], uint8, range [0, 255].
    """
    if images.shape[-3:-1] == (height, width):
        return np.asarray(images, dtype=np.uint8)

    original_shape = images.shape
    images_flat = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_single(np.asarray(im, dtype=np.uint8), height, width) for im in images_flat])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])
