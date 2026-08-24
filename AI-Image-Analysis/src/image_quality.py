"""Transparent heuristic image quality calculations."""
from __future__ import annotations
import cv2
import numpy as np
from .image_analysis import to_gray


def assess_quality(image: np.ndarray) -> dict[str, float | str]:
    gray = to_gray(image)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    noise = float(np.median(np.abs(gray.astype(np.float32) - cv2.GaussianBlur(gray, (3, 3), 0))) * 1.4826)
    brightness, contrast = float(gray.mean()), float(gray.std())
    dynamic_range = float(np.percentile(gray, 99) - np.percentile(gray, 1))
    sharp_score = min(sharpness / 5, 100); contrast_score = min(contrast / 0.64, 100)
    brightness_score = max(0, 100 - abs(brightness - 127.5) / 127.5 * 100)
    noise_score = max(0, 100 - noise * 5)
    score = float(np.clip(.35 * sharp_score + .25 * contrast_score + .2 * brightness_score + .2 * noise_score, 0, 100))
    label = "Excellent" if score >= 80 else "Good" if score >= 60 else "Moderate" if score >= 40 else "Poor"
    return {"noise_level": round(noise, 3), "sharpness": round(sharpness, 3), "blur_score": round(1 / (sharpness + 1) * 1000, 3), "brightness": round(brightness, 3), "contrast": round(contrast, 3), "dynamic_range": round(dynamic_range, 3), "quality_score": round(score, 1), "classification": label}
