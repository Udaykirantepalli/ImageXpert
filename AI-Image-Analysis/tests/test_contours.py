from src.contours import analyze_contours
def test_contours(rgb_image):
    annotated, data = analyze_contours(rgb_image, 10); assert annotated.shape == rgb_image.shape and data["number_of_contours"] >= 1
def test_no_contours():
    import numpy as np
    _, data = analyze_contours(np.zeros((100, 100), dtype=np.uint8)); assert data["number_of_contours"] == 0
