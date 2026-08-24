"""Non-destructive, parameter-bounded preprocessing operations."""
from __future__ import annotations
import cv2
import numpy as np
from .image_analysis import to_gray


def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if not 32 <= width <= 4096 or not 32 <= height <= 4096:
        raise ValueError("Resize dimensions must be between 32 and 4096 pixels.")
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA if width * height < image.shape[0] * image.shape[1] else cv2.INTER_CUBIC)


def normalize(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def equalize(image: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(to_gray(image))


def clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    if not 1.0 <= clip_limit <= 4.0:
        raise ValueError("CLAHE clip limit must be between 1.0 and 4.0.")
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(to_gray(image))


def enhance_contrast(image: np.ndarray, alpha: float = 1.2, beta: int = 0) -> np.ndarray:
    if not 1.0 <= alpha <= 2.0 or not -50 <= beta <= 50:
        raise ValueError("Contrast settings are outside the safe range.")
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
