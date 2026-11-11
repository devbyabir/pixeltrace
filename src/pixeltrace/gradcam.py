"""Grad-CAM: where in the image does the model look when it says "AI"?

Works for CNNs (4-D feature maps) and ViTs (token sequences reshaped to a grid).

    python -m pixeltrace.gradcam --run results/runs/resnet50 --n 12
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import CLASS_NAMES
from .data import SplitDataset
from .inference import load_run
from .models import gradcam_target_layer


class GradCAM:
    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module):
        self.model, self.acts, self.grads = model, None, None
        self._h = [layer.register_forward_hook(self._fwd)]

    def _fwd(self, _m, _i, out):
        out = out[0] if isinstance(out, (tuple, list)) else out
        self.acts = out
        if out.requires_grad:  # skip during torch.no_grad() forward passes
            out.register_hook(lambda g: setattr(self, "grads", g))

    @staticmethod
    def _to_grid(t: torch.Tensor) -> torch.Tensor:
        """(B, N, C) tokens -> (B, C, H, W), dropping class/register tokens."""
        if t.dim() == 4:
            return t
        n = t.shape[1]
        side = int(math.isqrt(n))
        extra = n - side * side
        return t[:, extra:, :].transpose(1, 2).reshape(t.shape[0], t.shape[2], side, side)

    def __call__(self, x: torch.Tensor, target: int = 1) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        logits[:, target].sum().backward()
        a, g = self._to_grid(self.acts), self._to_grid(self.grads)
        cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.detach().cpu().numpy()

    def close(self):
        for h in self._h:
            h.remove()


def run(run_dir: str | Path, n: int = 12, seed: int = 0) -> Path:
    run_dir = Path(run_dir)
    model, cfg, tf, device = load_run(run_dir)
    ds = SplitDataset(cfg.data_dir, "test", tf, limit=cfg.limit)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(ds), size=min(n, len(ds)), replace=False)

    cam = GradCAM(model, gradcam_target_layer(model))
    cols = 4
    rows = math.ceil(len(picks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.3), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), picks):
        x, y, _ = ds[int(i)]
        x = x.unsqueeze(0).to(device).requires_grad_(True)
        heat = cam(x, target=1)[0]
        with torch.no_grad():
            p_ai = torch.softmax(model(x), 1)[0, 1].item()
        img = Image.open(ds.data_dir / ds.df.iloc[int(i)]["path"]).convert("RGB").resize((cfg.img_size, cfg.img_size))
        ax.imshow(img)
        ax.imshow(heat, cmap="jet", alpha=0.4)
        ax.set_title(f"true {CLASS_NAMES[y]}  P(ai)={p_ai:.2f}", fontsize=9)
    cam.close()
    fig.suptitle(f"Grad-CAM for class 'ai': {run_dir.name}")
    fig.tight_layout()
    out = run_dir / "gradcam.jpg"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    run(a.run, a.n)


if __name__ == "__main__":
    main()
