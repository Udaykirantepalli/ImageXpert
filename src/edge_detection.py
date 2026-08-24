"""Bounded multi-algorithm edge detection and comparison metrics."""
from __future__ import annotations
from time import perf_counter
import cv2
import numpy as np
from skimage.filters import roberts, prewitt
from .image_analysis import to_gray

EDGE_ALGORITHMS = ("Roberts", "Prewitt", "Sobel", "Laplacian", "Canny")
CANNY_PRESETS = {"Low": (50, 100), "Balanced": (75, 150), "High": (100, 200)}


def detect_edges(image: np.ndarray, algorithm: str, canny_preset: str = "Balanced", sobel_kernel: int = 3) -> tuple[np.ndarray, dict[str, float | int]]:
    """Run a known algorithm and return its binary display image plus metrics."""
    if algorithm not in EDGE_ALGORITHMS:
        raise ValueError("Unsupported edge detector.")
    if sobel_kernel not in (3, 5, 7):
        raise ValueError("Sobel kernel must be 3, 5, or 7.")
    gray = to_gray(image)
    started = perf_counter()
    if algorithm == "Roberts":
        raw = roberts(gray.astype(np.float32) / 255.0)
        strength = np.clip(raw * 255, 0, 255).astype(np.uint8)
        edges = (strength > 25).astype(np.uint8) * 255
    elif algorithm == "Prewitt":
        raw = prewitt(gray.astype(np.float32) / 255.0)
        strength = np.clip(raw * 255, 0, 255).astype(np.uint8)
        edges = (strength > 25).astype(np.uint8) * 255
    elif algorithm == "Sobel":
        x, y = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel)
        strength = cv2.convertScaleAbs(cv2.magnitude(x, y))
        edges = (strength > 50).astype(np.uint8) * 255
    elif algorithm == "Laplacian":
        strength = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        edges = (strength > 35).astype(np.uint8) * 255
    else:
        if canny_preset not in CANNY_PRESETS:
            raise ValueError("Unsupported Canny preset.")
        edges = cv2.Canny(gray, *CANNY_PRESETS[canny_preset])
        strength = edges
    elapsed = (perf_counter() - started) * 1000
    count = int(np.count_nonzero(edges))
    return edges, {"edge_pixels": count, "edge_density": round(count / edges.size * 100, 4), "processing_time_ms": round(elapsed, 3), "relative_edge_strength": round(float(strength.mean()) / 255 * 100, 3)}


def compare_edges(image: np.ndarray, canny_preset: str = "Balanced", sobel_kernel: int = 3) -> tuple[dict[str, np.ndarray], list[dict[str, float | int | str]]]:
    images, metrics = {}, []
    for algorithm in EDGE_ALGORITHMS:
        output, values = detect_edges(image, algorithm, canny_preset, sobel_kernel)
        images[algorithm] = output
        metrics.append({"algorithm": algorithm, **values})
    return images, metrics
