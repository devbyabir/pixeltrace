"""Train one model on the fixed train split, select on val, report once on test.

    python -m pixeltrace.train --config configs/resnet50.yaml
    python -m pixeltrace.train --config configs/simple_cnn.yaml --epochs 2 --limit 64   # smoke test
"""
from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from .common import environment_info, get_device, load_config, seed_everything
from .data import build_loader
from .metrics import (bootstrap_ci, compute_metrics, error_analysis, plot_confusion, plot_history,
                      plot_roc, predict, save_json)
from .models import build_model, normalization


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mirror(files, backup_dir: Path | None) -> None:
    """Copy files to the persistent backup folder right away (atomic rename, so a disconnect never
    leaves a half-written checkpoint behind)."""
    if backup_dir is None:
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.exists():
            tmp = backup_dir / (f.name + ".tmp")
            shutil.copy2(f, tmp)
            tmp.replace(backup_dir / f.name)


def run(cfg, device_name: str | None = None, backup_dir: str | None = None) -> dict:
    backup_dir = Path(backup_dir) if backup_dir else None
    seed_everything(cfg.seed)
    device = get_device(device_name)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")
    save_json(environment_info(), run_dir / "environment.json")
    print(f"[{cfg.model}] device={device} -> {run_dir}")

    model = build_model(cfg.model, cfg.pretrained, drop_rate=cfg.drop_rate).to(device)
    mean, std = normalization(model)
    common = dict(img_size=cfg.img_size, mean=mean, std=std, batch_size=cfg.batch_size,
                  num_workers=cfg.num_workers, limit=cfg.limit)
    train_dl = build_loader(cfg.data_dir, "train", train=True, **common)
    val_dl = build_loader(cfg.data_dir, "val", **common)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * len(train_dl)
    sched = cosine_with_warmup(opt, cfg.warmup_epochs * len(train_dl), steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auc, best_epoch, history = -1.0, -1, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y, _ in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(y)
            n += len(y)

        p, yv, _ = predict(model, val_dl, device)
        vm = compute_metrics(yv, p)
        val_loss = float(nn.functional.binary_cross_entropy(
            torch.tensor(p).clamp(1e-6, 1 - 1e-6), torch.tensor(yv, dtype=torch.float32)))
        row = {"epoch": epoch, "train_loss": total / n, "val_loss": val_loss,
               "val_accuracy": vm["accuracy"], "val_roc_auc": vm.get("roc_auc", float("nan")),
               "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

        # model selection on the validation split only (the old notebooks used the test split here)
        score = vm.get("roc_auc", vm["accuracy"])
        if score > best_auc:
            best_auc, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "config": cfg.__dict__, "epoch": epoch}, run_dir / "best.pt")
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            mirror([run_dir / "best.pt", run_dir / "history.csv", run_dir / "config.json",
                    run_dir / "environment.json"], backup_dir)
            print(f"  saved best checkpoint (epoch {epoch}){' + backed up' if backup_dir else ''}", flush=True)
        elif epoch - best_epoch >= cfg.patience:
            print(f"  early stop: no val improvement for {cfg.patience} epochs")
            break

    hist = pd.DataFrame(history)
    hist.to_csv(run_dir / "history.csv", index=False)
    plot_history(hist, run_dir / "history.png")

    # ---- single, final look at the untouched test split with the best checkpoint
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=False)["model"])
    test_dl = build_loader(cfg.data_dir, "test", **common)
    p, yt, idx = predict(model, test_dl, device)
    metrics = compute_metrics(yt, p)
    metrics.update(bootstrap_ci(yt, p))
    metrics.update({"model": cfg.model, "run": run_dir.name, "best_epoch": best_epoch,
                    "epochs_run": len(history), "smoke": bool(cfg.limit),
                    "best_val_roc_auc": round(best_auc, 4),
                    "params_m": round(sum(q.numel() for q in model.parameters()) / 1e6, 2)})
    save_json(metrics, run_dir / "metrics.json")
    plot_confusion(metrics, run_dir / "confusion_matrix.png", f"{run_dir.name} (test)")
    plot_roc(yt, p, run_dir / "roc_curve.png", run_dir.name)
    error_analysis(test_dl.dataset.df, yt, p, idx, test_dl.dataset.data_dir, run_dir)
    print(f"[{run_dir.name}] TEST acc={metrics['accuracy']:.4f} macroF1={metrics['macro_f1']:.4f} "
          f"AUC={metrics.get('roc_auc', float('nan')):.4f}  (best epoch {best_epoch})")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None, help="cuda / cpu / mps (default: auto)")
    ap.add_argument("--backup-dir", default=None, help="copy best.pt there on every improvement")
    for k, t in [("epochs", int), ("batch_size", int), ("lr", float), ("limit", int),
                 ("num_workers", int), ("seed", int), ("run_name", str), ("data_dir", str),
                 ("output_dir", str)]:
        ap.add_argument(f"--{k.replace('_', '-')}", dest=k, type=t, default=None)
    args = vars(ap.parse_args())
    config, device, backup_dir = args.pop("config"), args.pop("device"), args.pop("backup_dir")
    run(load_config(config, args), device, backup_dir)


if __name__ == "__main__":
    main()
