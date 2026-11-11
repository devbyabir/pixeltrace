"""How much does test performance drop under everyday image degradations?

Social media re-compresses and resizes images, so a detector that only works on pristine files
is of limited use. Evaluates a trained run on the test split under each corruption.

    python -m pixeltrace.robustness --run results/runs/resnet50
"""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageFilter

from .data import SplitDataset
from .inference import load_run
from .metrics import compute_metrics, predict
from .preprocess import jpeg_roundtrip


def downscale(im: Image.Image, factor: float) -> Image.Image:
    w, h = im.size
    small = im.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.Resampling.BILINEAR)
    return small.resize((w, h), Image.Resampling.BILINEAR)


def blur(im: Image.Image, radius: float) -> Image.Image:
    return im.filter(ImageFilter.GaussianBlur(radius))


CORRUPTIONS = {
    "clean": [("-", None)],
    "jpeg": [(q, partial(jpeg_roundtrip, quality=q)) for q in (90, 70, 50, 30)],
    "downscale": [(f, partial(downscale, factor=f)) for f in (0.75, 0.5, 0.25)],
    "blur": [(r, partial(blur, radius=r)) for r in (0.5, 1.0, 2.0)],
}


def run(run_dir: str | Path, batch_size: int = 64, num_workers: int = 2) -> pd.DataFrame:
    from torch.utils.data import DataLoader

    run_dir = Path(run_dir)
    model, cfg, tf, device = load_run(run_dir)
    rows = []
    for family, levels in CORRUPTIONS.items():
        for level, fn in levels:
            ds = SplitDataset(cfg.data_dir, "test", tf, corrupt=fn, limit=cfg.limit)
            dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
            p, y, _ = predict(model, dl, device)
            m = compute_metrics(y, p)
            rows.append({"corruption": family, "level": level, "accuracy": m["accuracy"],
                         "roc_auc": m.get("roc_auc"), "ai_recall": m["ai_recall"],
                         "real_recall": m["real_recall"]})
            print(f"  {family:>9} {str(level):>5}  acc={m['accuracy']:.3f}  auc={m.get('roc_auc', float('nan')):.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "robustness.csv", index=False)

    clean_auc = df.loc[df.corruption == "clean", "roc_auc"].iloc[0]
    fams = [f for f in CORRUPTIONS if f != "clean"]
    fig, axes = plt.subplots(1, len(fams), figsize=(4 * len(fams), 3.4), sharey=True)
    for ax, fam in zip(axes, fams):
        d = df[df.corruption == fam]
        ax.plot([str(v) for v in d.level], d.roc_auc, "o-")
        ax.axhline(clean_auc, ls="--", color="grey", lw=1, label="clean")
        ax.axhline(0.5, ls=":", color="red", lw=1, label="chance")
        ax.set_title(fam)
        ax.set_xlabel({"jpeg": "quality", "downscale": "factor", "blur": "radius"}[fam])
    axes[0].set_ylabel("test ROC-AUC")
    axes[0].legend()
    fig.suptitle(f"Robustness: {run_dir.name}")
    fig.tight_layout()
    fig.savefig(run_dir / "robustness.png", dpi=200)
    plt.close(fig)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    a = ap.parse_args()
    run(a.run, a.batch_size, a.num_workers)


if __name__ == "__main__":
    main()
