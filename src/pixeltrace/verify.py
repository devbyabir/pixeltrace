"""Release verification: fails loudly unless every required run and artifact is present and consistent.

Checks per run: artifacts exist; not a smoke run; checkpoint reload reproduces the saved test
predictions; raw-file inference (pixeltrace.preprocess.normalise on the original file) matches the
prediction on the pre-processed copy; metrics recomputed from the saved predictions match
metrics.json. Also checks deployment.json was chosen by validation and the ONNX parity report.

    python -m pixeltrace.verify --require simple_cnn resnet50 efficientnet_b0 --raw-dir data/raw
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .inference import load_run, predict_images
from .metrics import compute_metrics, predict
from .data import build_loader
from .models import normalization

RUN_ARTIFACTS = ["best.pt", "config.json", "environment.json", "history.csv", "history.png",
                 "metrics.json", "confusion_matrix.png", "roc_curve.png", "test_predictions.csv",
                 "robustness.csv", "robustness.png", "gradcam.jpg", "errors_grid.jpg"]


def check(cond: bool, msg: str, failures: list[str]) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def verify_run(run_dir: Path, raw_dir: Path | None, n_raw: int, tol: float, failures: list[str],
               allow_smoke: bool = False) -> dict:
    print(f"== {run_dir.name}")
    missing = [a for a in RUN_ARTIFACTS if not (run_dir / a).exists()]
    check(not missing, f"artifacts present {'' if not missing else missing}", failures)
    if "best.pt" in missing or "test_predictions.csv" in missing:
        return {}
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if not allow_smoke:
        check(not metrics.get("smoke"), "full run (not a smoke run)", failures)

    # checkpoint reload -> identical test predictions
    model, cfg, _, device = load_run(run_dir)
    mean, std = normalization(model)
    dl = build_loader(cfg.data_dir, "test", cfg.img_size, mean, std, 64, 2, limit=cfg.limit)
    p, y, idx = predict(model, dl, device)
    saved = pd.read_csv(run_dir / "test_predictions.csv").set_index("id")
    ids = dl.dataset.df.iloc[idx]["id"].to_numpy()
    diff = np.abs(saved.loc[ids, "p_ai"].to_numpy() - p)
    check(diff.max() <= tol, f"reload reproduces saved test predictions (max |dp|={diff.max():.2e})", failures)

    # metrics recomputed from saved predictions match metrics.json
    m2 = compute_metrics(saved["label"].to_numpy(), saved["p_ai"].to_numpy())
    same = all(abs(m2[k] - metrics[k]) < 1e-3 for k in ("accuracy", "macro_f1", "roc_auc"))
    check(same, "metrics.json consistent with test_predictions.csv", failures)
    check("accuracy_95ci" in metrics and "roc_auc_95ci" in metrics, "bootstrap CIs present", failures)
    check(metrics["n"] == len(saved), f"test size {metrics['n']} matches predictions", failures)

    # raw original file -> same score as the pre-processed copy
    out = {"run": run_dir.name, "reload_max_diff": float(diff.max())}
    if raw_dir is not None:
        sample = saved.sort_values("id").head(n_raw)
        preds = predict_images(run_dir, [raw_dir / s for s in sample["source_file"]])
        rd = np.abs(np.array([r["p_ai"] for r in preds]) - sample["p_ai"].to_numpy())
        # predict_images rounds to 4 decimals
        check(rd.max() <= max(tol, 1e-4), f"raw-file inference matches processed copy on {len(rd)} images "
              f"(max |dp|={rd.max():.2e})", failures)
        out["raw_inference_max_diff"] = float(rd.max())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--require", nargs="+", required=True)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--n-raw", type=int, default=20)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--onnx-report", default="results/onnx/onnx_parity.json")
    ap.add_argument("--deployment", default="results/deployment.json")
    ap.add_argument("--allow-smoke", action="store_true")
    ap.add_argument("--out", default="results/verification.json")
    a = ap.parse_args()

    failures: list[str] = []
    runs = [verify_run(Path(a.runs) / r, Path(a.raw_dir) if a.raw_dir else None, a.n_raw, a.tol, failures,
                       a.allow_smoke) for r in a.require]

    print("== deployment / onnx")
    dep_path = Path(a.deployment)
    if dep_path.exists():
        dep = json.loads(dep_path.read_text(encoding="utf-8"))
        best = max(dep["candidates"], key=dep["candidates"].get)
        check(dep["selection_metric"] == "best_val_roc_auc" and dep["run"] == best,
              f"deployment model '{dep['run']}' is the validation-best run", failures)
    else:
        check(False, "deployment.json present", failures)
    onnx = Path(a.onnx_report)
    if onnx.exists():
        rep = json.loads(onnx.read_text(encoding="utf-8"))
        check(rep.get("passed", False), f"ONNX parity (max |dp|={rep.get('torch_vs_onnx_max_abs_diff')})", failures)
    else:
        check(False, "ONNX parity report present", failures)

    result = {"passed": not failures, "smoke_allowed": a.allow_smoke, "failures": failures, "runs": runs,
              "torch": torch.__version__, "cuda": torch.cuda.is_available()}
    Path(a.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nVERIFICATION", "PASSED" if not failures else f"FAILED ({len(failures)} checks)")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
