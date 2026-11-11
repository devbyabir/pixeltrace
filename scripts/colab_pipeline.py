"""End-to-end pipeline for a GPU machine (written for Google Colab, works anywhere).

Every step runs with check=True: the first failing command stops the pipeline with a non-zero exit.

    python scripts/colab_pipeline.py --mode smoke                         # 1 epoch, 64 images/split
    python scripts/colab_pipeline.py --mode full --backup /content/drive/MyDrive/pixeltrace_backup

Smoke runs write to results/smoke/ and can never be mixed into results/runs/ (compare/verify
refuse smoke runs there). Full mode REQUIRES --backup (e.g. a mounted Google Drive folder): the
pipeline writes, reads back and deletes a probe file there first and stops if that fails. During
training the best checkpoint is copied to the backup on every improvement; each finished model and
the final reports are copied as soon as they exist. A rerun restores finished models from the backup.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"simple_cnn": "configs/simple_cnn.yaml",
          "resnet50": "configs/resnet50.yaml",
          "efficientnet_b0": "configs/efficientnet_b0.yaml"}
RAW_GDRIVE_ID = os.environ.get("PIXELTRACE_RAW_GDRIVE_ID", "")  # private dataset link: not published (see DATASET.md)


def run(cmd: list[str], log: Path | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # stream child progress line by line
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env=env)
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    rc = proc.wait()
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"$ {' '.join(cmd)}\n{''.join(lines)}\n[exit {rc}, {time.time() - t0:.0f}s]\n\n")
    if rc != 0:
        raise SystemExit(f"FAILED (exit {rc}): {' '.join(cmd)}")


def py(*args: str, log: Path | None = None) -> None:
    run([sys.executable, *args], log)


def backup(src: Path, backup_root: Path | None, name: str) -> None:
    if backup_root is None:
        return
    dst = backup_root / name
    # overwrite in place (never delete the previous backup first), then confirm every file arrived
    shutil.copytree(src, dst, dirs_exist_ok=True)
    bad = [f.relative_to(src) for f in src.rglob("*") if f.is_file()
           and (not (dst / f.relative_to(src)).exists() or (dst / f.relative_to(src)).stat().st_size != f.stat().st_size)]
    if bad:
        raise SystemExit(f"Backup of {src} incomplete: {bad[:5]}")
    print(f"  backed up + size-checked {src} -> {dst}", flush=True)


def check_backup(root: Path) -> None:
    """Fail fast unless the backup folder is really writable and readable (e.g. Drive is mounted)."""
    import uuid
    root.mkdir(parents=True, exist_ok=True)
    probe = root / f".write_probe_{uuid.uuid4().hex}"
    token = uuid.uuid4().hex
    probe.write_text(token, encoding="utf-8")
    ok = probe.read_text(encoding="utf-8") == token
    probe.unlink()
    if not ok:
        raise SystemExit(f"Backup folder {root} failed the write/read check")
    print(f"  backup folder OK (write/read probe passed): {root}", flush=True)


def complete(run_dir: Path) -> bool:
    need = ["best.pt", "metrics.json", "test_predictions.csv", "robustness.csv", "gradcam.jpg"]
    return all((run_dir / f).exists() for f in need)


def ensure_data(raw: Path, log: Path) -> None:
    if not (raw / "AI Generated").is_dir():
        zip_path = ROOT / "data" / "raw_dataset.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if not zip_path.exists():
            if not RAW_GDRIVE_ID:
                raise SystemExit("No raw data: put it in data/raw/ or set PIXELTRACE_RAW_GDRIVE_ID (see DATASET.md)")
            run(["gdown", "-q", RAW_GDRIVE_ID, "-O", str(zip_path)], log)
        tmp = ROOT / "data" / "_unzip"
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        src = next(p.parent for p in tmp.rglob("AI Generated") if p.is_dir())
        raw.mkdir(parents=True, exist_ok=True)
        for cls in ("AI Generated", "Real"):
            shutil.move(str(src / cls), str(raw / cls))
        shutil.rmtree(tmp)
    # rebuild the cleaned data + manifest and prove it matches the committed split
    check_dir = ROOT / "results" / "_rebuild_check"
    py("scripts/prepare_data.py", "--raw", str(raw), "--out", "data/processed",
       "--report-dir", str(check_dir), log=log)
    py("scripts/verify_manifest.py", "--reference-digest", "results/manifest/digest.json",
       "--candidate", str(check_dir / "manifest" / "splits_manifest.csv"), log=log)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["smoke", "full"], required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--backup", default=None, help="folder (e.g. on Google Drive) for per-model backups")
    ap.add_argument("--skip-data", action="store_true", help="data/processed already prepared and verified")
    a = ap.parse_args()

    smoke = a.mode == "smoke"
    if not smoke and not a.backup:
        raise SystemExit("--mode full requires --backup <persistent folder> (refusing to train without a backup)")
    base = ROOT / "results" / ("smoke" if smoke else "")
    runs_dir = base / "runs"
    log = base / "logs" / f"pipeline_{a.mode}.log"
    backup_root = Path(a.backup) / a.mode if a.backup else None
    if backup_root:
        check_backup(backup_root)
    started = time.strftime("%Y-%m-%d %H:%M:%S")

    # environment snapshot
    env_dir = base / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "pip_freeze.txt").write_text(
        subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout,
        encoding="utf-8")
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        smi_text = smi.stdout or smi.stderr
    except FileNotFoundError:  # CPU-only machine
        smi_text = "nvidia-smi not found: no NVIDIA GPU on this machine\n"
    (env_dir / "nvidia_smi.txt").write_text(smi_text, encoding="utf-8")

    if not a.skip_data:
        ensure_data(ROOT / a.raw, log)

    extra = ["--epochs", "1", "--limit", "64"] if smoke else []
    for name in a.models:
        run_dir = runs_dir / name
        restore = backup_root / "runs" / name if backup_root else None
        if not smoke and restore and complete(restore) and not complete(run_dir):
            shutil.copytree(restore, run_dir, dirs_exist_ok=True)
            print(f"  restored finished run {name} from backup; not retraining", flush=True)
        if complete(run_dir) and not smoke:
            print(f"  {name}: already complete", flush=True)
        else:
            bk = ["--backup-dir", str(backup_root / "runs" / name)] if backup_root else []
            py("-m", "pixeltrace.train", "--config", MODELS[name], "--output-dir", str(runs_dir), *extra, *bk, log=log)
            py("-m", "pixeltrace.robustness", "--run", str(run_dir), log=log)
            py("-m", "pixeltrace.gradcam", "--run", str(run_dir), log=log)
        backup(run_dir, backup_root, f"runs/{name}")

    # compare -> deployment choice (validation) -> ONNX export + parity -> verification
    smoke_flag = ["--allow-smoke"] if smoke else []
    py("-m", "pixeltrace.compare", "--runs", str(runs_dir), "--out", str(base),
       "--require", *a.models, *smoke_flag, log=log)
    deployment = json.loads((base / "deployment.json").read_text(encoding="utf-8"))
    py("scripts/make_reference_images.py", "--out", "docs/samples", log=log)
    py("-m", "pixeltrace.export_onnx", "--run", str(runs_dir / deployment["run"]), "--out", str(base / "onnx"),
       "--reference-images", "docs/samples", log=log)
    py("-m", "pixeltrace.verify", "--runs", str(runs_dir), "--require", *a.models, "--raw-dir", a.raw,
       "--deployment", str(base / "deployment.json"), "--onnx-report", str(base / "onnx" / "onnx_parity.json"),
       "--out", str(base / "verification.json"), *smoke_flag, log=log)

    (base / "pipeline_run.json").write_text(json.dumps({
        "mode": a.mode, "models": a.models, "started": started,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "deployment": deployment["run"]}, indent=2), encoding="utf-8")
    if backup_root:
        for item in ("summary.md", "summary.csv", "summary.png", "deployment.json", "verification.json",
                     "pipeline_run.json", "onnx", "environment", "logs"):
            p = base / item
            if p.is_dir():
                backup(p, backup_root, item)
            elif p.exists():
                shutil.copy2(p, backup_root / item)
    print(f"\nPIPELINE {a.mode.upper()} OK -> {base}")


if __name__ == "__main__":
    main()
