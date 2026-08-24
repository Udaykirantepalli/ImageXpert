"""Safe in-memory image decoding and normalization."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = False
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 25_000_000
MAX_DIMENSION = 4096
ALLOWED_FORMATS = {"JPEG", "PNG"}
# Pillow raises on images beyond twice this limit before allocating decoded pixels.
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class ImageUploadError(ValueError):
    """A safe, user-presentable image upload error."""


@dataclass(frozen=True)
class LoadedImage:
    """A decoded RGB/RGBA/grayscale image and safe source metadata."""
    array: np.ndarray
    format: str
    was_resized: bool


def load_image(data: bytes) -> LoadedImage:
    """Decode a JPEG or PNG safely without trusting the supplied filename."""
    if not data:
        raise ImageUploadError("Please select a non-empty JPG or PNG image.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageUploadError("This image file is too large to process safely (maximum 15 MB).")
    try:
        with Image.open(BytesIO(data)) as probe:
            image_format = probe.format
            if image_format not in ALLOWED_FORMATS:
                raise ImageUploadError("Only valid JPG and PNG images are supported.")
            width, height = probe.size
            if width < 1 or height < 1 or width * height > MAX_PIXELS:
                raise ImageUploadError("This image is too large to process safely. Please upload a smaller image.")
            probe.verify()
        with Image.open(BytesIO(data)) as decoded:
            decoded.load()
            # Preserve supported channel semantics while avoiding palette/CMYK surprises.
            mode = "RGBA" if "A" in decoded.getbands() else ("L" if decoded.mode == "L" else "RGB")
            decoded = decoded.convert(mode)
            was_resized = max(decoded.size) > MAX_DIMENSION
            if was_resized:
                decoded.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)
            return LoadedImage(np.asarray(decoded).copy(), image_format, was_resized)
    except ImageUploadError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, SyntaxError) as exc:
        raise ImageUploadError("The uploaded image appears to be corrupted. Please try another image.") from exc
