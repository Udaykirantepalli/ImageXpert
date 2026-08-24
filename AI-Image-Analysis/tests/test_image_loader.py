import pytest
from .conftest import image_bytes
from src.image_loader import ImageUploadError, MAX_UPLOAD_BYTES, load_image

def test_valid_png_and_jpeg():
    assert load_image(image_bytes("RGB", "PNG")).format == "PNG"
    assert load_image(image_bytes("RGB", "JPEG")).array.shape[2] == 3
def test_corrupted_empty_and_unsupported():
    for payload in (b"", b"broken", b"GIF89a"):
        with pytest.raises(ImageUploadError): load_image(payload)
def test_oversized_upload():
    with pytest.raises(ImageUploadError): load_image(b"x" * (MAX_UPLOAD_BYTES + 1))
def test_rgba_and_grayscale():
    assert load_image(image_bytes("RGBA")).array.shape[2] == 4
    assert load_image(image_bytes("L")).array.ndim == 2

def test_rejects_excessive_dimensions():
    from io import BytesIO
    from PIL import Image
    stream = BytesIO(); Image.new("L", (5001, 5000)).save(stream, format="PNG")
    with pytest.raises(ImageUploadError): load_image(stream.getvalue())
