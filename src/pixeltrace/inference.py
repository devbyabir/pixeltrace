"""Load a trained run and classify new images.

    python -m pixeltrace.inference --run results/runs/resnet50 photo1.jpg photo2.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from .common import Config, get_device
from .data import build_transforms
from .models import build_model, normalization
from .preprocess import normalise


def load_run(run_dir: str | Path, device: torch.device | None = None):
    """Returns (model in eval mode, Config, eval transform, device)."""
    run_dir = Path(run_dir)
    device = device or get_device()
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    cfg = Config(**ckpt["config"])
    # pretrained=False: the fine-tuned weights come from the checkpoint, no download needed
    model = build_model(cfg.model, pretrained=False, drop_rate=cfg.drop_rate)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    mean, std = normalization(model)
    return model, cfg, build_transforms(cfg.img_size, mean, std, train=False), device


@torch.no_grad()
def predict_images(run_dir, paths) -> list[dict]:
    model, _, tf, device = load_run(run_dir)
    out = []
    for p in paths:
        img = normalise(Image.open(p))  # same cleaning as the training data
        p_ai = torch.softmax(model(tf(img).unsqueeze(0).to(device)), dim=1)[0, 1].item()
        out.append({"file": str(p), "p_ai": p_ai, "prediction": "ai" if p_ai >= 0.5 else "real"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory containing best.pt")
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()
    for r in predict_images(args.run, args.images):
        print(f"{r['prediction']:>4}  P(ai)={r['p_ai']:.3f}  {r['file']}")


if __name__ == "__main__":
    main()
