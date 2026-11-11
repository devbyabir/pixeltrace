"""Write Pillow reference vectors for the JS port of the resampler (docs/assets/preprocess.js).

For each synthetic sample image: the RGB pixels after to_rgb(), Pillow's ImageOps.fit(512, LANCZOS)
output, the JPEG-q95 round-tripped 512 image, and Pillow's BILINEAR 224 output of that. The Node test
feeds the *same input pixels* to the JS code, so any difference is a bug in the port.

    python tools/parity/make_vectors.py --samples docs/samples --out dist/parity
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pixeltrace.preprocess import jpeg_roundtrip, square_resize, to_rgb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--samples", default="docs/samples")
ap.add_argument("--out", default="dist/parity")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
index = []
for p in sorted(Path(a.samples).iterdir()):
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    rgb = to_rgb(Image.open(p))
    sq = square_resize(rgb, 512)
    jp = jpeg_roundtrip(sq, 95)
    small = jp.resize((224, 224), Image.Resampling.BILINEAR)
    stem = p.stem
    for name, im in (("rgb", rgb), ("sq", sq), ("jpg", jp), ("small", small)):
        np.asarray(im, dtype=np.uint8).tofile(out / f"{stem}.{name}.bin")
    tensor = (np.asarray(small, dtype=np.float32) / 255.0).transpose(2, 0, 1).copy()
    sha = lambda im: hashlib.sha256(np.asarray(im, dtype=np.uint8).tobytes()).hexdigest()
    index.append({"name": stem, "file": p.name, "w": rgb.width, "h": rgb.height,
                  "sq_sha256": sha(sq), "jpg_sha256": sha(jp), "small_sha256": sha(small),
                  "tensor_sha256": hashlib.sha256(tensor.tobytes()).hexdigest()})
(out / "index.json").write_text(json.dumps(index, indent=2))
print(f"{len(index)} samples -> {out}")
