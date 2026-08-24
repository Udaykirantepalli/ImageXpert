"""Vectorized descriptive image statistics."""
from __future__ import annotations
import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return an 8-bit grayscale copy for gray, RGB, or RGBA input."""
    if image.ndim == 2:
        return image.copy()
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Expected a grayscale, RGB, or RGBA image.")
    return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY if image.shape[2] == 4 else cv2.COLOR_RGB2GRAY)


def analyze_image(image: np.ndarray) -> dict[str, object]:
    """Compute compact, serializable image and histogram statistics."""
    gray = to_gray(image)
    height, width = gray.shape
    channels = 1 if image.ndim == 2 else image.shape[2]
    hist = np.bincount(gray.ravel(), minlength=256)
    probabilities = hist[hist > 0] / gray.size
    result: dict[str, object] = {
        "width": int(width), "height": int(height), "channels": int(channels),
        "total_pixels": int(gray.size), "aspect_ratio": round(width / height, 4),
        "mean_brightness": float(gray.mean()), "brightness_std": float(gray.std()),
        "contrast": float(gray.std()), "min_pixel_value": int(gray.min()), "max_pixel_value": int(gray.max()),
        "grayscale_mean": float(gray.mean()), "grayscale_std": float(gray.std()),
        "entropy": float(-(probabilities * np.log2(probabilities)).sum()), "histogram": hist.tolist(),
    }
    if image.ndim == 3:
        rgb = image[:, :, :3].astype(np.float32)
        result["rgb_channel_means"] = [float(v) for v in rgb.mean(axis=(0, 1))]
        result["rgb_channel_stds"] = [float(v) for v in rgb.std(axis=(0, 1))]
    else:
        result["rgb_channel_means"] = [float(gray.mean())] * 3
        result["rgb_channel_stds"] = [float(gray.std())] * 3
    return result
