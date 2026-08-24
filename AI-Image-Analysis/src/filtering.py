"""Safe fixed-kernel noise-reduction filters."""
from __future__ import annotations
import cv2
import numpy as np
ALLOWED_KERNELS = (3, 5, 7)


def apply_filter(image: np.ndarray, filter_name: str, kernel_size: int = 5) -> np.ndarray:
    if kernel_size not in ALLOWED_KERNELS:
        raise ValueError("Kernel size must be one of 3, 5, or 7.")
    operations = {
        "Mean": lambda: cv2.blur(image, (kernel_size, kernel_size)),
        "Gaussian": lambda: cv2.GaussianBlur(image, (kernel_size, kernel_size), 0),
        "Median": lambda: cv2.medianBlur(image, kernel_size),
        "Bilateral": lambda: cv2.bilateralFilter(image, kernel_size, 75, 75),
    }
    if filter_name not in operations:
        raise ValueError("Unsupported filter.")
    return operations[filter_name]()


def all_filters(image: np.ndarray, kernel_size: int = 5) -> dict[str, np.ndarray]:
    return {name: apply_filter(image, name, kernel_size) for name in ("Mean", "Gaussian", "Median", "Bilateral")}
