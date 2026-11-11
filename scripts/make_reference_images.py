"""Generate synthetic, non-dataset reference images for preprocessing / ONNX parity checks.

These are procedurally drawn test patterns (no photos, no dataset content), so they can be
published with the website. They deliberately cover the awkward cases of the pipeline:
portrait / landscape / square, PNG with alpha, palette PNG, tiny images, EXIF rotation.

    python scripts/make_reference_images.py --out docs/samples
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def gradient(w: int, h: int) -> Image.Image:
    x = np.linspace(0, 1, w)[None, :, None]
    y = np.linspace(0, 1, h)[:, None, None]
    rgb = np.concatenate([np.broadcast_to(x, (h, w, 1)), np.broadcast_to(y, (h, w, 1)),
                          np.broadcast_to((x + y) / 2, (h, w, 1))], axis=2)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def shapes(w: int, h: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    im = gradient(w, h)
    d = ImageDraw.Draw(im)
    for _ in range(25):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        r = int(rng.integers(10, max(11, min(w, h) // 4)))
        col = tuple(int(c) for c in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            d.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=col)
        else:
            d.rectangle([x0 - r, y0 - r, x0 + r, y0 + r // 2], fill=col)
    return im


def noise(w: int, h: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = np.asarray(gradient(w, h), dtype=np.float32)
    return Image.fromarray(np.clip(base + rng.normal(0, 25, base.shape), 0, 255).astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/samples")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    shapes(1200, 700, 1).save(out / "pattern_landscape.jpg", quality=90)
    noise(600, 900, 2).save(out / "pattern_portrait_noise.jpg", quality=85)
    shapes(1024, 1024, 3).save(out / "pattern_square.png")

    rgba = shapes(800, 600, 4).convert("RGBA")
    a = np.asarray(rgba).copy()
    a[:, : a.shape[1] // 3, 3] = 0          # fully transparent band
    a[:, a.shape[1] // 3: 2 * a.shape[1] // 3, 3] = 255
    a[:, 2 * a.shape[1] // 3:, 3] = 255     # (partial alpha is avoided: canvas premultiplies it)
    Image.fromarray(a, "RGBA").save(out / "pattern_alpha.png")

    shapes(640, 480, 5).convert("P", palette=Image.Palette.ADAPTIVE, colors=64).save(out / "pattern_palette.png")
    shapes(150, 100, 6).save(out / "pattern_tiny.jpg", quality=92)

    rot = shapes(900, 600, 7)
    exif = Image.Exif()
    exif[0x0112] = 6                        # orientation: rotate 90° CW on display
    rot.save(out / "pattern_exif_rotated.jpg", quality=90, exif=exif)
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
