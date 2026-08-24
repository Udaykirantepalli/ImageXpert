"""Contour detection on a separate annotated image copy."""
from __future__ import annotations
import cv2
import numpy as np
from .image_analysis import to_gray


def analyze_contours(image: np.ndarray, min_area: int = 100) -> tuple[np.ndarray, dict[str, object]]:
    if not 10 <= min_area <= 10000:
        raise ValueError("Minimum contour area must be between 10 and 10000.")
    gray = to_gray(image)
    edges = cv2.Canny(gray, 75, 150)
    found, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [contour for contour in found if cv2.contourArea(contour) >= min_area]
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else image[:, :, :3].copy()
    objects = []
    for contour in valid:
        area = float(cv2.contourArea(contour)); perimeter = float(cv2.arcLength(contour, True)); x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])) if moments["m00"] else (x + w // 2, y + h // 2)
        objects.append({"area": round(area, 2), "perimeter": round(perimeter, 2), "bounding_rectangle": (x, y, w, h), "center": center})
        cv2.drawContours(canvas, [contour], -1, (0, 255, 0), 2); cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 80, 0), 1)
    largest = max(objects, key=lambda item: item["area"], default=None)
    return canvas, {"number_of_contours": len(valid), "approximate_object_count": len(valid), "largest_contour": largest, "objects": objects}
