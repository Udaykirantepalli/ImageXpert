import numpy as np
from src.image_analysis import analyze_image, to_gray
from src.preprocessing import clahe, equalize, normalize, resize_image
def test_analysis_rgb_and_gray(rgb_image):
    stats = analyze_image(rgb_image); assert stats["width"] == 160 and stats["channels"] == 3 and stats["entropy"] >= 0
    assert analyze_image(to_gray(rgb_image))["channels"] == 1
def test_preprocessing(rgb_image):
    assert resize_image(rgb_image, 80, 60).shape[:2] == (60, 80)
    assert normalize(rgb_image).dtype == np.uint8
    assert equalize(rgb_image).ndim == 2 and clahe(rgb_image).ndim == 2
