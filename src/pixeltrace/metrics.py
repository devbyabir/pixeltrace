"""Probability-based metrics, plots and error analysis."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from . import CLASS_NAMES


@torch.no_grad()
def predict(model: torch.nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (P(ai) per image, true labels, dataset row indices)."""
    model.eval()
    probs, labels, idxs = [], [], []
    for x, y, i in loader:
        logits = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
        labels.append(y)
        idxs.append(i)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy(), torch.cat(idxs).numpy()


def compute_metrics(y: np.ndarray, p_ai: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (p_ai >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = {
        "n": int(len(y)),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "ai_precision": precision_score(y, pred, pos_label=1, zero_division=0),
        "ai_recall": recall_score(y, pred, pos_label=1, zero_division=0),
        "ai_f1": f1_score(y, pred, pos_label=1, zero_division=0),
        "real_recall": recall_score(y, pred, pos_label=0, zero_division=0),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": threshold,
    }
    # ranking metrics need both classes present
    if len(np.unique(y)) == 2:
        m["roc_auc"] = roc_auc_score(y, p_ai)
        m["average_precision"] = average_precision_score(y, p_ai)
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}


def bootstrap_ci(y: np.ndarray, p_ai: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    """95% bootstrap confidence intervals: with a 300-image test set these are not negligible."""
    rng = np.random.default_rng(seed)
    acc, auc = [], []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        if len(np.unique(y[s])) < 2:
            continue
        acc.append(accuracy_score(y[s], p_ai[s] >= 0.5))
        auc.append(roc_auc_score(y[s], p_ai[s]))
    q = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    return {"accuracy_95ci": q(acc), "roc_auc_95ci": q(auc)}


def plot_confusion(m: dict, path: Path, title: str = "") -> None:
    cm = m["confusion_matrix"]
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(mat, cmap="Blues")
    for (r, c), v in np.ndenumerate(mat):
        ax.text(c, r, str(v), ha="center", va="center", fontsize=14,
                color="white" if v > mat.max() / 2 else "black")
    ax.set_xticks([0, 1], [f"pred {n}" for n in CLASS_NAMES])
    ax.set_yticks([0, 1], [f"true {n}" for n in CLASS_NAMES])
    ax.set_title(title or "Confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roc(y: np.ndarray, p_ai: np.ndarray, path: Path, label: str = "") -> None:
    fpr, tpr, _ = roc_curve(y, p_ai)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot(fpr, tpr, lw=2, label=f"{label} AUC={roc_auc_score(y, p_ai):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (AI recall)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_history(history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="val accuracy")
    axes[1].plot(history["epoch"], history["val_roc_auc"], label="val ROC-AUC")
    axes[1].set_title("Validation")
    for a in axes:
        a.set_xlabel("epoch")
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def error_analysis(df: pd.DataFrame, y: np.ndarray, p_ai: np.ndarray, idx: np.ndarray,
                   data_dir: Path, out_dir: Path, k: int = 16) -> pd.DataFrame:
    """Save every test prediction and a grid of the most confident mistakes."""
    rows = df.iloc[idx].copy()
    rows["p_ai"] = p_ai
    rows["pred"] = (p_ai >= 0.5).astype(int)
    rows["correct"] = rows["pred"] == y
    rows["confidence_wrong"] = np.where(rows["correct"], 0.0, np.abs(p_ai - 0.5) * 2)
    rows.to_csv(out_dir / "test_predictions.csv", index=False)

    wrong = rows[~rows["correct"]].sort_values("confidence_wrong", ascending=False).head(k)
    if len(wrong):
        cols = 4
        nrows = int(np.ceil(len(wrong) / cols))
        fig, axes = plt.subplots(nrows, cols, figsize=(cols * 3, nrows * 3.3), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (_, r) in zip(axes.ravel(), wrong.iterrows()):
            ax.imshow(Image.open(data_dir / r["path"]).convert("RGB"))
            ax.set_title(f"true {CLASS_NAMES[r['label']]}  P(ai)={r['p_ai']:.2f}\n{r['orig_width']}x{r['orig_height']} {r['orig_format']}",
                         fontsize=8)
        fig.suptitle("Most confident mistakes (test)")
        fig.tight_layout()
        fig.savefig(out_dir / "errors_grid.jpg", dpi=110)
        plt.close(fig)
    return rows


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
