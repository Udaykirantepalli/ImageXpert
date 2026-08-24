"""Transparent, editable rule-based recommendations (not a trained model)."""
from __future__ import annotations


def generate_recommendations(analysis: dict[str, object], quality: dict[str, object], edge_density: float) -> list[dict[str, str]]:
    brightness, contrast, noise, sharpness = float(analysis["mean_brightness"]), float(analysis["contrast"]), float(quality["noise_level"]), float(quality["sharpness"])
    filter_name = "Gaussian" if noise > 8 else "Median" if noise > 4 else "No filter needed"
    contrast_action = "CLAHE contrast enhancement" if contrast < 35 or brightness < 70 or brightness > 190 else "Keep current contrast"
    detector = "Canny" if noise < 10 else "Sobel"
    sharpen = "Consider sharpening" if sharpness < 80 else "Sharpness is sufficient"
    sufficient = "Yes" if float(quality["quality_score"]) >= 60 else "No — improve before downstream analysis"
    pipeline = "Grayscale → " + (filter_name if filter_name != "No filter needed" else "No filter") + " → " + contrast_action + f" → {detector} → Contour Detection"
    return [
        {"recommendation": f"Noise reduction: {filter_name}", "reason": f"Estimated noise level is {noise:.1f}.", "confidence": "High" if noise > 8 else "Moderate", "suggested_pipeline": pipeline},
        {"recommendation": contrast_action, "reason": f"Brightness is {brightness:.0f} and contrast is {contrast:.0f}.", "confidence": "High" if contrast < 35 else "Moderate", "suggested_pipeline": pipeline},
        {"recommendation": sharpen, "reason": f"Laplacian sharpness is {sharpness:.0f}.", "confidence": "Moderate", "suggested_pipeline": pipeline},
        {"recommendation": f"Recommended edge detector: {detector}", "reason": f"Observed edge density is {edge_density:.2f}%.", "confidence": "High" if edge_density < 25 else "Moderate", "suggested_pipeline": pipeline},
        {"recommendation": f"Image quality sufficient: {sufficient}", "reason": f"Heuristic quality score: {quality['quality_score']}/100 ({quality['classification']}).", "confidence": "Moderate", "suggested_pipeline": pipeline},
    ]
