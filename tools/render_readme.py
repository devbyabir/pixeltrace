"""Insert the real results table into README.md between the RESULTS markers.

    python tools/render_readme.py
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"simple_cnn": "Simple CNN (from scratch)", "resnet50": "ResNet-50 (ImageNet-pretrained)",
          "efficientnet_b0": "EfficientNet-B0 (ImageNet-pretrained)"}


def f(v, d=3):
    return f"{float(v):.{d}f}"


def ci(s):
    a = json.loads(s.replace("'", '"')) if isinstance(s, str) else s
    return f"{a[0]:.3f}–{a[1]:.3f}"


def main():
    res = ROOT / "results"
    rows = list(csv.DictReader(open(res / "summary.csv", encoding="utf-8")))
    dep = json.loads((res / "deployment.json").read_text(encoding="utf-8"))
    ver = json.loads((res / "verification.json").read_text(encoding="utf-8"))
    onnx = json.loads((res / "onnx" / "onnx_parity.json").read_text(encoding="utf-8"))
    env = json.loads((res / "runs" / dep["run"] / "environment.json").read_text(encoding="utf-8"))
    lines = [
        "Test split: 300 images, evaluated once with the checkpoint chosen on validation. "
        f"Deployment model, chosen by validation ROC-AUC: **{LABELS.get(dep['run'], dep['run'])}**.",
        "",
        "| Model | Params (M) | Val ROC-AUC | Test accuracy (95 % CI) | Macro-F1 | AI precision / recall | Test ROC-AUC (95 % CI) | ROC-AUC @ JPEG q50 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {LABELS.get(r['run'], r['run'])} | {f(r['params_m'], 1)} | {f(r['best_val_roc_auc'])} | "
            f"{f(r['accuracy'])} ({ci(r['accuracy_95ci'])}) | {f(r['macro_f1'])} | {f(r['ai_precision'])} / {f(r['ai_recall'])} | "
            f"{f(r['roc_auc'])} ({ci(r['roc_auc_95ci'])}) | {f(r['roc_auc_jpeg50']) if r.get('roc_auc_jpeg50') else '–'} |")
    lines += [
        "",
        "![Validation vs test](results/summary.png)",
        "",
        f"Trained with PyTorch {env['packages'].get('torch')} / timm {env['packages'].get('timm')} on "
        f"{env.get('gpu') or 'CPU'}. Verification (`results/verification.json`): "
        f"**{'all checks passed' if ver['passed'] else 'FAILED'}**. "
        f"ONNX vs PyTorch on the test split: max |ΔP(AI)| = {onnx['torch_vs_onnx_max_abs_diff']:.1e}.",
        "",
        "Per-model training curves, confusion matrices, ROC curves, robustness curves and per-image test predictions "
        "are in `results/runs/<model>/`. Grad-CAM overlays and error grids contain dataset images and are not published.",
    ]
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    new = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->",
                 "<!-- RESULTS:START -->\n" + "\n".join(lines) + "\n<!-- RESULTS:END -->", text, flags=re.S)
    readme.write_text(new, encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
