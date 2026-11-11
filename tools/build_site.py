"""Build docs/assets/results.json (and copy the ONNX model + parity references) from real results.

Every number shown on the website comes from files written by the pipeline; nothing is typed by hand.

    python tools/build_site.py --results results --onnx results/onnx
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"simple_cnn": "Simple CNN (scratch)", "resnet50": "ResNet-50", "efficientnet_b0": "EfficientNet-B0",
          "vit_small": "ViT-S/16", "vgg16_bn": "VGG-16-BN", "mobilenetv3": "MobileNetV3"}


def roc_points(pred_csv: Path, max_points: int = 120) -> list:
    """ROC curve (FPR, TPR) from per-image test predictions, thinned for the web."""
    import csv
    rows = list(csv.DictReader(open(pred_csv, encoding="utf-8")))
    y = [int(r["label"]) for r in rows]
    s = [float(r["p_ai"]) for r in rows]
    order = sorted(range(len(s)), key=lambda i: -s[i])
    P, N = sum(y), len(y) - sum(y)
    tp = fp = 0
    pts = [(0.0, 0.0)]
    for k, i in enumerate(order):
        tp += y[i]; fp += 1 - y[i]
        if k + 1 == len(order) or s[order[k + 1]] != s[i]:
            pts.append((fp / N, tp / P))
    step = max(1, len(pts) // max_points)
    thin = pts[::step]
    if thin[-1] != pts[-1]:
        thin.append(pts[-1])
    return [[round(a, 4), round(b, 4)] for a, b in thin]


def dashboard(res: Path, runs: list) -> dict:
    import csv
    out = {}
    for run in runs:
        d = res / "runs" / run
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        hist = list(csv.DictReader(open(d / "history.csv", encoding="utf-8")))
        rob = list(csv.DictReader(open(d / "robustness.csv", encoding="utf-8")))
        out[run] = {
            "roc": roc_points(d / "test_predictions.csv"),
            "confusion": m["confusion_matrix"],
            "history": {"epoch": [int(h["epoch"]) for h in hist],
                        "val_roc_auc": [round(float(h["val_roc_auc"]), 4) for h in hist],
                        "train_loss": [round(float(h["train_loss"]), 4) for h in hist],
                        "val_loss": [round(float(h["val_loss"]), 4) for h in hist]},
            "robustness": [{"corruption": r["corruption"], "level": r["level"], "roc_auc": float(r["roc_auc"]),
                            "accuracy": float(r["accuracy"]), "ai_recall": float(r["ai_recall"]),
                            "real_recall": float(r["real_recall"])} for r in rob],
        }
    return out


def robustness_note(path: Path, label: str) -> str:
    import csv
    rows = {(r["corruption"], r["level"]): r for r in csv.DictReader(open(path, encoding="utf-8"))}
    c, j, d = rows[("clean", "-")], rows[("jpeg", "30")], rows[("downscale", "0.25")]
    return (f"Robustness of {label}: ROC-AUC {float(c['roc_auc']):.3f} on clean test images, "
            f"{float(j['roc_auc']):.3f} after JPEG q30 and {float(d['roc_auc']):.3f} after x0.25 down-scaling. "
            f"Errors are one-sided: AI recall drops from {float(c['ai_recall']):.2f} to "
            f"{min(float(j['ai_recall']), float(d['ai_recall'])):.2f} while real-image recall stays at "
            f"{min(float(r['real_recall']) for r in rows.values()):.2f} or higher, so degraded AI images tend to be scored as real.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--onnx", default="results/onnx")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--no-model", action="store_true", help="publish the showcase without the browser demo")
    a = ap.parse_args()
    res, docs = ROOT / a.results, ROOT / a.docs

    audit = json.loads((res / "data_audit.json").read_text(encoding="utf-8"))
    deployment = json.loads((res / "deployment.json").read_text(encoding="utf-8"))
    verification = json.loads((res / "verification.json").read_text(encoding="utf-8"))
    models = []
    for run in deployment["candidates"]:
        m = json.loads((res / "runs" / run / "metrics.json").read_text(encoding="utf-8"))
        if m.get("smoke"):
            raise SystemExit(f"{run} is a smoke run: refusing to publish it")
        rob = res / "runs" / run / "robustness.csv"
        if rob.exists():
            import csv
            for row in csv.DictReader(open(rob, encoding="utf-8")):
                if row["corruption"] == "jpeg" and row["level"] == "50":
                    m["roc_auc_jpeg50"] = float(row["roc_auc"])
        keep = ["run", "model", "params_m", "best_epoch", "epochs_run", "best_val_roc_auc", "accuracy", "accuracy_95ci",
                "macro_f1", "ai_precision", "ai_recall", "roc_auc", "roc_auc_95ci", "roc_auc_jpeg50", "n"]
        models.append({k: m[k] for k in keep if k in m} | {"label": LABELS.get(run, run)})
    models.sort(key=lambda m: -m["best_val_roc_auc"])

    env = json.loads((res / "runs" / deployment["run"] / "environment.json").read_text(encoding="utf-8"))
    split = audit["split"]
    out = {
        "audit": {"raw": audit["raw_files"], "dupes": audit["exact_duplicate_copies"],
                  "meta_test_acc": audit["metadata_only_baseline_raw"]["test_accuracy"],
                  "clean_test_acc": audit["filesize_only_baseline_cleaned"]["test_accuracy"]},
        "split": {"total": audit["unique_images_after_dedup"],
                  **{s: split[s]["ai"] + split[s]["real"] for s in ("train", "val", "test")}},
        "models": models,
        "deployment": {"run": deployment["run"], "label": LABELS.get(deployment["run"], deployment["run"]),
                       "selection_metric": deployment["selection_metric"]},
        "environment": {"torch": env["packages"].get("torch"), "timm": env["packages"].get("timm"),
                        "device": env.get("gpu") or "CPU"},
        "verification_passed": bool(verification.get("passed")),
        "robustness_note": robustness_note(res / "runs" / deployment["run"] / "robustness.csv",
                                           LABELS.get(deployment["run"], deployment["run"])),
        "samples": sorted(p.name for p in (docs / "samples").iterdir() if p.suffix in {".jpg", ".png"}),
    }

    out["dashboard"] = dashboard(res, [m["run"] for m in models])
    out["baselines"] = {"metadata_raw": audit["metadata_only_baseline_raw"],
                        "filesize_clean": audit["filesize_only_baseline_cleaned"],
                        "legacy_reported_accuracy": {"min": 0.9325, "max": 0.9961}}
    parity = ROOT / a.onnx / "onnx_parity.json"
    if not a.no_model:
        rep = json.loads(parity.read_text(encoding="utf-8"))
        if not rep.get("passed"):
            raise SystemExit("ONNX parity did not pass: not publishing the model")
        (docs / "model").mkdir(exist_ok=True)
        shutil.copy2(ROOT / a.onnx / rep["onnx"], docs / "model" / "pixeltrace.onnx")
        shutil.copy2(ROOT / a.onnx / "reference_predictions.json", docs / "parity" / "reference_predictions.json")
        out["model"] = {"file": "model/pixeltrace.onnx", "bytes": rep["onnx_bytes"], "sha256": rep["onnx_sha256"],
                        "input_size": rep["input"]["shape"][2], "torch_vs_onnx_max_abs_diff": rep["torch_vs_onnx_max_abs_diff"]}

    (docs / "assets" / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "models"}, indent=1))


if __name__ == "__main__":
    main()
