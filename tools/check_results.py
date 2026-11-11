"""Independent local check of a downloaded results/ folder (NumPy + pandas only, no PyTorch).

Recomputes accuracy / macro-F1 / ROC-AUC from every run's test_predictions.csv, compares them with
metrics.json, checks the test ids against the committed split manifest, and confirms the Colab-side
verification and ONNX parity reports passed.

    python tools/check_results.py --results results
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["simple_cnn", "resnet50", "efficientnet_b0"]
ARTIFACTS = ["config.json", "environment.json", "history.csv", "history.png", "metrics.json",
             "confusion_matrix.png", "roc_curve.png", "test_predictions.csv", "robustness.csv", "robustness.png"]


def auc(y, s):
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = y == 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def macro_f1(y, p):
    f = []
    for c in (0, 1):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        f.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    res = Path(ap.parse_args().results)
    fails = []

    def check(ok, msg):
        print(("PASS  " if ok else "FAIL  ") + msg)
        if not ok:
            fails.append(msg)

    man = pd.read_csv(ROOT / "results" / "manifest" / "splits_manifest.csv").set_index("id")
    test_ids = set(man.index[man.split == "test"])
    for run in REQUIRED:
        d = res / "runs" / run
        missing = [a for a in ARTIFACTS if not (d / a).exists()]
        check(not missing, f"{run}: artifacts present {missing or ''}")
        if missing:
            continue
        m = json.loads((d / "metrics.json").read_text())
        check(not m.get("smoke"), f"{run}: full run (not smoke)")
        pr = pd.read_csv(d / "test_predictions.csv")
        check(set(pr["id"]) == test_ids, f"{run}: predictions cover exactly the 300 manifest test ids")
        lab = pr["id"].map(man["label"]).to_numpy()
        check((lab == pr["label"].to_numpy()).all(), f"{run}: labels match the manifest")
        y, s = pr["label"].to_numpy(), pr["p_ai"].to_numpy()
        p = (s >= 0.5).astype(int)
        rec = {"accuracy": (p == y).mean(), "macro_f1": macro_f1(y, p), "roc_auc": auc(y, s)}
        ok = all(abs(rec[k] - m[k]) < 1e-3 for k in rec)
        check(ok, f"{run}: recomputed acc={rec['accuracy']:.4f} f1={rec['macro_f1']:.4f} auc={rec['roc_auc']:.4f} "
                  f"match metrics.json")
        env = json.loads((d / "environment.json").read_text())
        check(bool(env.get("cuda_available")), f"{run}: trained on GPU ({env.get('gpu')})")
        cfg = json.loads((d / "config.json").read_text())
        check(cfg.get("limit", 0) == 0, f"{run}: no --limit sub-sampling")

    dep = json.loads((res / "deployment.json").read_text())
    best = max(dep["candidates"], key=dep["candidates"].get)
    check(dep["run"] == best and dep["selection_metric"] == "best_val_roc_auc",
          f"deployment '{dep['run']}' = best validation ROC-AUC")
    ver = json.loads((res / "verification.json").read_text())
    check(ver.get("passed") and not ver.get("smoke_allowed"), "Colab verification.json passed (full mode)")
    onnx = json.loads((res / "onnx" / "onnx_parity.json").read_text())
    check(onnx.get("passed") and onnx["run"] == dep["run"],
          f"ONNX parity passed for deployment model (max |dp|={onnx['torch_vs_onnx_max_abs_diff']:.1e})")
    import hashlib
    cands = [res / "onnx" / onnx["onnx"], ROOT / "docs" / "model" / "pixeltrace.onnx"]
    found = [c for c in cands if c.exists()]
    check(bool(found) and all(hashlib.sha256(c.read_bytes()).hexdigest() == onnx["onnx_sha256"] for c in found),
          f"ONNX file present with the hash from onnx_parity.json ({', '.join(str(c.relative_to(ROOT)) for c in found)})")
    print("\nLOCAL CHECK", "PASSED" if not fails else f"FAILED ({len(fails)})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
