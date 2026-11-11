"""Export a trained run to ONNX and check PyTorch vs ONNX Runtime parity.

The exported graph takes RGB in [0, 1], shape (N, 3, S, S), applies the backbone's own mean/std
normalisation internally, and returns softmax scores (N, 2) = [P(real), P(ai)]. A browser client
therefore only has to reproduce the pixel pipeline (pixeltrace.preprocess.normalise + resize).

    python -m pixeltrace.export_onnx --run results/runs/efficientnet_b0 --out results/onnx \
        --reference-images docs/samples
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from .data import SplitDataset
from .inference import load_run
from .models import normalization
from .preprocess import normalise


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model((x - self.mean) / self.std), dim=1)


def pixel_tensor(img: Image.Image, size: int) -> np.ndarray:
    """Python reference for the browser: normalise() -> PIL bilinear resize -> float32 CHW in [0,1].
    Equivalent to torchvision Resize(size)+CenterCrop(size)+ToTensor on the square 512 image."""
    sq = normalise(img)
    small = sq.resize((size, size), Image.Resampling.BILINEAR)
    return (np.asarray(small, dtype=np.float32) / 255.0).transpose(2, 0, 1).copy()


def export(model: nn.Module, size: int, path: Path) -> None:
    dummy = torch.rand(1, 3, size, size)
    kwargs = dict(input_names=["pixels"], output_names=["scores"], opset_version=17,
                  dynamic_axes={"pixels": {0: "batch"}, "scores": {0: "batch"}})
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:  # older torch without the `dynamo` argument
        torch.onnx.export(model, dummy, str(path), **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="results/onnx")
    ap.add_argument("--reference-images", default=None, help="folder of non-dataset images for browser parity")
    ap.add_argument("--test-limit", type=int, default=0, help="0 = whole test split")
    ap.add_argument("--tolerance", type=float, default=1e-4)
    a = ap.parse_args()

    import onnxruntime as ort

    run_dir, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, cfg, _, _ = load_run(run_dir, torch.device("cpu"))
    mean, std = normalization(model)
    wrapped = ExportWrapper(model, mean, std).eval()
    onnx_path = out / "pixeltrace.onnx"
    export(wrapped, cfg.img_size, onnx_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # 1) model parity on the whole test split (same tensors into both runtimes)
    from torchvision import transforms as T
    tf = T.Compose([T.Resize(cfg.img_size), T.CenterCrop(cfg.img_size), T.ToTensor()])
    ds = SplitDataset(cfg.data_dir, "test", tf, limit=a.test_limit)
    diffs, agree = [], 0
    with torch.no_grad():
        for start in range(0, len(ds), 32):
            x = torch.stack([ds[i][0] for i in range(start, min(start + 32, len(ds)))])
            pt = wrapped(x)[:, 1].numpy()
            ox = sess.run(["scores"], {"pixels": x.numpy()})[0][:, 1]
            diffs.append(np.abs(pt - ox))
            agree += int(((pt >= 0.5) == (ox >= 0.5)).sum())
    d = np.concatenate(diffs)
    report = {
        "run": run_dir.name, "model": cfg.model, "onnx": onnx_path.name,
        "onnx_bytes": onnx_path.stat().st_size,
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "input": {"name": "pixels", "shape": [None, 3, cfg.img_size, cfg.img_size], "range": "[0,1] RGB"},
        "output": {"name": "scores", "columns": ["real", "ai"]},
        "test_images": int(len(d)),
        "torch_vs_onnx_max_abs_diff": float(d.max()), "torch_vs_onnx_mean_abs_diff": float(d.mean()),
        "label_agreement": agree / len(d), "tolerance": a.tolerance, "passed": bool(d.max() <= a.tolerance),
    }

    # 2) reference outputs for the browser on non-dataset images (full raw-file pipeline)
    if a.reference_images:
        refs = []
        for p in sorted(Path(a.reference_images).iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            x = pixel_tensor(Image.open(p), cfg.img_size)
            with torch.no_grad():
                pt = float(wrapped(torch.from_numpy(x)[None])[0, 1])
            ox = float(sess.run(["scores"], {"pixels": x[None]})[0][0, 1])
            refs.append({"file": p.name, "file_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                         "p_ai_torch": round(pt, 6), "p_ai_onnx": round(ox, 6),
                         "tensor_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                         "tensor_mean": float(x.mean())})
        report["reference_images"] = refs
        (out / "reference_predictions.json").write_text(json.dumps(refs, indent=2), encoding="utf-8")

    (out / "onnx_parity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "reference_images"}, indent=2))
    if not report["passed"]:
        raise SystemExit(f"ONNX parity FAILED: max diff {d.max():.2e} > {a.tolerance}")


if __name__ == "__main__":
    main()
