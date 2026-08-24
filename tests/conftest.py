from io import BytesIO
import numpy as np
from PIL import Image
import pytest

@pytest.fixture
def rgb_image():
    image = np.zeros((120, 160, 3), dtype=np.uint8); image[25:95, 35:125] = (220, 100, 40)
    return image

def image_bytes(mode="RGB", fmt="PNG", size=(40, 30)):
    image = Image.new(mode, size, 120)
    stream = BytesIO(); image.save(stream, format=fmt); return stream.getvalue()
