import pytest
from src.filtering import all_filters, apply_filter
def test_filters(rgb_image):
    results = all_filters(rgb_image, 5); assert set(results) == {"Mean", "Gaussian", "Median", "Bilateral"}
    assert all(result.shape == rgb_image.shape for result in results.values())
def test_rejects_unbounded_kernel(rgb_image):
    with pytest.raises(ValueError): apply_filter(rgb_image, "Mean", 9)
