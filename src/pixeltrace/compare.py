"""Collect every run's metrics.json into one table and pick the deployment model.

The deployment model is chosen by **validation** ROC-AUC (ties -> fewer parameters).
Test metrics are reported but never used for the choice.

    python -m pixeltrace.compare            # writes results/summary.{md,csv,png} + results/deployment.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLS = ["run", "model", "params_m", "best_epoch", "epochs_run", "best_val_roc_auc",
        "accuracy", "accuracy_95ci", "macro_f1", "ai_precision", "ai_recall", "roc_auc", "roc_auc_95ci"]


def load_runs(runs_dir: Path, required: list[str] | None = None, allow_smoke: bool = False) -> pd.DataFrame:
    rows = []
    for f in sorted(runs_dir.glob("*/metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        if m.get("smoke") and not allow_smoke:
            raise SystemExit(f"{f} is a smoke run; keep smoke runs out of {runs_dir}")
        rob = f.parent / "robustness.csv"
        if rob.exists():
            r = pd.read_csv(rob)
            for q in (70, 50):
                sel = r[(r.corruption == "jpeg") & (r.level.astype(str) == str(q))]
                if len(sel):
                    m[f"roc_auc_jpeg{q}"] = round(float(sel.roc_auc.iloc[0]), 4)
        rows.append(m)
    if not rows:
        raise SystemExit(f"No metrics.json under {runs_dir}")
    df = pd.DataFrame(rows)
    missing = sorted(set(required or []) - set(df["run"]))
    if missing:
        raise SystemExit(f"Missing required runs: {missing}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--require", nargs="*", default=[], help="run names that must be present")
    ap.add_argument("--allow-smoke", action="store_true", help="pipeline test only; never for reported results")
    a = ap.parse_args()

    df = load_runs(Path(a.runs), a.require, a.allow_smoke)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- deployment choice: validation only
    ranked = df.sort_values(["best_val_roc_auc", "params_m"], ascending=[False, True])
    chosen = ranked.iloc[0]
    deployment = {
        "run": chosen["run"], "model": chosen["model"],
        "selection_metric": "best_val_roc_auc", "selection_value": float(chosen["best_val_roc_auc"]),
        "candidates": {r["run"]: float(r["best_val_roc_auc"]) for _, r in ranked.iterrows()},
        "note": "Chosen on the validation split only; test metrics were not used for this choice.",
    }
    (out / "deployment.json").write_text(json.dumps(deployment, indent=2), encoding="utf-8")

    cols = [c for c in COLS + ["roc_auc_jpeg70", "roc_auc_jpeg50"] if c in df.columns]
    table = df[cols].sort_values("best_val_roc_auc", ascending=False)
    table.to_csv(out / "summary.csv", index=False)

    fmt = table.copy()
    for c in ("accuracy_95ci", "roc_auc_95ci"):
        if c in fmt:
            fmt[c] = fmt[c].apply(lambda v: f"{v[0]:.3f}–{v[1]:.3f}")
    header = "| " + " | ".join(fmt.columns) + " |\n|" + "---|" * len(fmt.columns) + "\n"
    body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in fmt.itertuples(index=False))
    (out / "summary.md").write_text(
        "# Results\n\nTest-split metrics (the test split was not used for checkpoint or model selection). "
        f"Rows sorted by validation ROC-AUC. Deployment model (by validation): **{chosen['run']}**.\n\n"
        + header + body + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(1.8 * len(table) + 2, 3.6))
    x = list(range(len(table)))
    ax.bar([i - 0.27 for i in x], table["best_val_roc_auc"], 0.27, label="val ROC-AUC")
    ax.bar(x, table["roc_auc"], 0.27, label="test ROC-AUC")
    ax.bar([i + 0.27 for i in x], table["accuracy"], 0.27, label="test accuracy")
    ax.set_xticks(x, table["run"], rotation=15)
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("PixelTrace: validation vs test")
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=200)
    print((out / "summary.md").read_text(encoding="utf-8"))
    print(f"deployment model (validation ROC-AUC): {chosen['run']}")


if __name__ == "__main__":
    main()
