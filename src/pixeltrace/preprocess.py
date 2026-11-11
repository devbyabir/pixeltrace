"""Image normalisation shared by dataset preparation and inference (no torch dependency)."""
from __future__ import annotations

import io

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None  # some real photos are ~100 MP


def to_rgb(im: Image.Image) -> Image.Image:
    """Apply EXIF rotation, flatten transparency onto white, convert to RGB."""
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        return bg
    return im.convert("RGB")


def square_resize(im: Image.Image, size: int) -> Image.Image:
    """Resize the shorter side to `size` (aspect ratio kept), then centre-crop to size x size."""
    return ImageOps.fit(im, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def jpeg_roundtrip(im: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def normalise(im: Image.Image, size: int = 512, quality: int = 95) -> Image.Image:
    """Exactly the cleaning applied to the training data: RGB, square crop, JPEG re-encode, no metadata."""
    return jpeg_roundtrip(square_resize(to_rgb(im), size), quality)
