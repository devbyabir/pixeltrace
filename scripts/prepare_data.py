"""Audit, de-duplicate, normalise and split the raw dataset.

Raw layout expected:   data/raw/<class name>/<image files>
Outputs:
    data/processed/images/<id>.jpg      normalised images (RGB, square, JPEG q=95, no metadata)
    data/processed/splits.csv           id, path, label, split, source file, ...
    results/data_audit.json / .md       audit numbers (duplicates, format/shape bias, shortcut-risk baseline)
    results/manifest/splits_manifest.csv  id, split, label, raw + processed SHA-256 (no pixels; safe to commit)

Why each step exists (all found in the original dataset):
  * 1,998 images were stored twice under different names -> exact + near-duplicate removal
    before splitting, so no detected duplicate pair is spread across train and test;
  * AI images were mostly square, often PNG and often carried EXIF, real images mostly
    non-square JPEGs -> every image is centre-cropped to a square, re-encoded as JPEG and
    stripped of metadata. This removes those particular cues; it does not prove that no
    other shortcut (e.g. source resolution / resampling statistics) remains;
  * a fixed, stratified, file-level split replaces the old per-epoch reshuffled batch split.

Usage:
    python scripts/prepare_data.py --raw data/raw --out data/processed
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pixeltrace.preprocess import square_resize, to_rgb  # noqa: E402  (also sets MAX_IMAGE_PIXELS)

# Folder name -> (label id, short name). Positive class (1) is "ai".
CLASS_MAP = {"AI Generated": (1, "ai"), "Real": (0, "real")}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# --------------------------------------------------------------------------- audit helpers
def dhash(img: Image.Image, size: int = 8) -> int:
    """64-bit difference hash, robust to resizing / re-compression."""
    g = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def scan(raw: Path) -> list[dict]:
    records = []
    for folder, (label, name) in CLASS_MAP.items():
        d = raw / folder
        if not d.is_dir():
            raise SystemExit(f"Missing class folder: {d}")
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            data = p.read_bytes()
            with Image.open(p) as im:
                w, h = im.size
                rec = {
                    "file": f"{folder}/{p.name}",
                    "label": label,
                    "class": name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "format": im.format,
                    "mode": im.mode,
                    "width": w,
                    "height": h,
                    "bytes": len(data),
                    "has_exif": bool(im.getexif()),
                    "dhash": dhash(im),
                }
            records.append(rec)
    return records


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def popcount64(x: np.ndarray) -> np.ndarray:
    """Number of set bits per uint64 (works on NumPy 1.x, which lacks np.bitwise_count)."""
    x = np.atleast_1d(np.ascontiguousarray(x, dtype=np.uint64))
    return np.unpackbits(x.view(np.uint8)).reshape(-1, 64).sum(1)


def group_duplicates(records: list[dict], max_hamming: int) -> list[list[int]]:
    """Groups of indices that are exact (sha256) or near (dHash) duplicates."""
    n = len(records)
    uf = UnionFind(n)
    first_by_sha: dict[str, int] = {}
    for i, r in enumerate(records):
        j = first_by_sha.setdefault(r["sha256"], i)
        if j != i:
            uf.union(i, j)
    if max_hamming >= 0:
        hashes = np.array([r["dhash"] for r in records], dtype=np.uint64)
        for i in range(n):
            dist = popcount64(hashes[i] ^ hashes[i + 1 :])
            for k in np.nonzero(dist <= max_hamming)[0]:
                uf.union(i, i + 1 + int(k))
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    return list(groups.values())


def bias_table(records: list[dict]) -> dict:
    out = {}
    for name in ("ai", "real"):
        rs = [r for r in records if r["class"] == name]
        n = len(rs)
        out[name] = {
            "images": n,
            "square_pct": round(100 * sum(r["width"] == r["height"] for r in rs) / n, 1),
            "png_pct": round(100 * sum(r["format"] == "PNG" for r in rs) / n, 1),
            "exif_pct": round(100 * sum(r["has_exif"] for r in rs) / n, 1),
            "median_megapixels": round(float(np.median([r["width"] * r["height"] / 1e6 for r in rs])), 2),
            "top_resolutions": [f"{w}x{h}: {c}" for (w, h), c in Counter((r["width"], r["height"]) for r in rs).most_common(3)],
        }
    return out


def metadata_features(r: dict) -> list[float]:
    return [
        r["width"] / r["height"],
        float(r["width"] == r["height"]),
        np.log(r["width"] * r["height"]),
        float(r["format"] == "PNG"),
        float(r["has_exif"]),
        np.log(r["bytes"]),
    ]


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney ROC-AUC with tie handling (no sklearn needed)."""
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_logreg(X: np.ndarray, y: np.ndarray, steps: int = 3000, lr: float = 0.1, l2: float = 1e-3):
    """Plain gradient-descent logistic regression with a quadratic feature expansion."""
    def expand(Z):
        return np.hstack([Z, Z ** 2, np.ones((len(Z), 1))])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xe = expand((X - mu) / sd)
    w = np.zeros(Xe.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Xe @ w))
        w -= lr * (Xe.T @ (p - y) / len(y) + l2 * w)
    return lambda Z: 1 / (1 + np.exp(-expand((Z - mu) / sd) @ w))


def shortcut_baseline(records: list[dict], featurize) -> dict:
    """Classifier on file metadata only (no pixels). High accuracy means the labels are predictable
    from non-content cues, i.e. a shortcut *risk* for image models. It does not show what any CNN learned.

    Same protocol as the image models: fitted on the train split only (fixed hyper-parameters,
    nothing tuned), reported on val and test.
    """
    def xy(split):
        rs = [r for r in records if r["split"] == split]
        return np.array([featurize(r) for r in rs], float), np.array([r["label"] for r in rs])

    Xtr, ytr = xy("train")
    model = fit_logreg(Xtr, ytr)
    out = {"fit_on": "train"}
    for split in ("val", "test"):
        X, y = xy(split)
        prob = model(X)
        out[f"{split}_accuracy"] = round(float(((prob >= 0.5) == y).mean()), 4)
        out[f"{split}_roc_auc"] = round(roc_auc(y, prob), 4)
    return out


# --------------------------------------------------------------------------- split
def stratified_split(records: list[dict], ratios: tuple[float, float, float], seed: int) -> None:
    rng = np.random.default_rng(seed)
    for label in (0, 1):
        idx = [i for i, r in enumerate(records) if r["label"] == label]
        rng.shuffle(idx)
        n = len(idx)
        n_tr, n_va = round(n * ratios[0]), round(n * ratios[1])
        for k, i in enumerate(idx):
            records[i]["split"] = "train" if k < n_tr else "val" if k < n_tr + n_va else "test"


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--report-dir", type=Path, default=Path("results"))
    ap.add_argument("--size", type=int, default=512, help="side of the square output images")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality of output images")
    ap.add_argument("--max-hamming", type=int, default=4, help="dHash distance for near-duplicates (-1 = exact only)")
    ap.add_argument("--split", type=float, nargs=3, default=(0.70, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Scanning {args.raw} ...")
    records = scan(args.raw)
    raw_bias = bias_table(records)

    # ---- de-duplication
    groups = group_duplicates(records, args.max_hamming)
    exact_dupes = len(records) - len({r["sha256"] for r in records})
    # how many distinct images occur once, twice, ... (by exact hash)
    multiplicity = Counter(Counter(r["sha256"] for r in records).values())
    keep, cross_class, cross_files = [], 0, 0
    for g in groups:
        labels = {records[i]["label"] for i in g}
        if len(labels) > 1:  # same picture in both classes: label is unreliable, drop all copies
            cross_class += 1
            cross_files += len(g)
            continue
        keep.append(records[min(g, key=lambda i: records[i]["file"])])
    keep.sort(key=lambda r: r["file"])
    near_dupes = len(records) - cross_files - len(keep) - exact_dupes
    print(f"  {len(records)} files -> {len(keep)} unique images "
          f"({exact_dupes} exact copies, ~{max(near_dupes, 0)} near-duplicates, "
          f"{cross_class} cross-class groups / {cross_files} files dropped)")

    # ---- split (file level, stratified, fixed seed)
    stratified_split(keep, tuple(args.split), args.seed)

    # ---- shortcut check on the *raw* files (before cleaning)
    raw_shortcut = shortcut_baseline(keep, metadata_features)
    print(f"  metadata-only classifier (fit on train) on RAW files: test acc {raw_shortcut['test_accuracy']:.3f}, "
          f"AUC {raw_shortcut['test_roc_auc']:.3f}")

    # ---- normalise images
    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, r in enumerate(keep):
        img_id = f"{r['class']}_{k:05d}"
        dst = img_dir / f"{img_id}.jpg"
        with Image.open(args.raw / r["file"]) as im:
            out = square_resize(to_rgb(im), args.size)
        out.save(dst, "JPEG", quality=args.quality, optimize=True)  # no exif passed -> stripped
        r["clean_bytes"] = dst.stat().st_size
        r["id"] = img_id
        r["clean_sha256"] = hashlib.sha256(dst.read_bytes()).hexdigest()
        rows.append({
            "id": img_id,
            "path": f"images/{img_id}.jpg",
            "label": r["label"],
            "class": r["class"],
            "split": r["split"],
            "source_file": r["file"],
            "orig_width": r["width"],
            "orig_height": r["height"],
            "orig_format": r["format"],
        })
        if (k + 1) % 250 == 0:
            print(f"  processed {k + 1}/{len(keep)}")

    with open(args.out / "splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Pixel-free manifest: lets anyone verify they rebuilt the same split from the same raw files.
    man_dir = args.report_dir / "manifest"
    man_dir.mkdir(parents=True, exist_ok=True)
    with open(man_dir / "splits_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "label", "source_file", "raw_sha256", "processed_sha256"])
        for r in keep:
            w.writerow([r["id"], r["split"], r["label"], r["file"], r["sha256"], r["clean_sha256"]])

    # After cleaning, the only metadata left is the JPEG file size (driven by image content).
    clean_shortcut = shortcut_baseline(keep, lambda r: [np.log(r["clean_bytes"])])
    print(f"  file-size-only classifier on CLEANED files: test acc {clean_shortcut['test_accuracy']:.3f}, "
          f"AUC {clean_shortcut['test_roc_auc']:.3f}")

    split_counts = Counter((r["split"], r["class"]) for r in keep)
    report = {
        "raw_files": len(records),
        "exact_duplicate_copies": exact_dupes,
        "distinct_images_by_occurrence": {str(k): v for k, v in sorted(multiplicity.items())},
        "near_duplicates_removed": max(near_dupes, 0),
        "unique_images_after_dedup": len(keep),
        "cross_class_duplicate_groups_dropped": cross_class,
        "near_duplicate_max_hamming": args.max_hamming,
        "raw_bias": raw_bias,
        "metadata_only_baseline_raw": raw_shortcut,
        "filesize_only_baseline_cleaned": clean_shortcut,
        "split_seed": args.seed,
        "split": {s: {c: split_counts[(s, c)] for c in ("ai", "real")} for s in ("train", "val", "test")},
        "output": {"size": args.size, "jpeg_quality": args.quality},
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "data_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.report_dir / "data_audit.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"Done. {len(rows)} images -> {img_dir}; report -> {args.report_dir / 'data_audit.md'}")


def render_markdown(r: dict) -> str:
    b = r["raw_bias"]
    lines = [
        "# Data audit",
        "",
        f"- Raw files: **{r['raw_files']}**",
        f"- Exact duplicate copies: **{r['exact_duplicate_copies']}** ("
        + "; ".join(f"{v} distinct images occur {k}x" for k, v in r["distinct_images_by_occurrence"].items()) + ")",
        f"- Unique images kept after exact + near-duplicate removal (dHash ≤ {r['near_duplicate_max_hamming']}): "
        f"**{r['unique_images_after_dedup']}**",
        f"- Cross-class duplicate groups dropped: {r['cross_class_duplicate_groups_dropped']}",
        "",
        "## Format / shape bias in the raw files (before de-duplication)",
        "",
        "| | AI | Real |",
        "|---|---|---|",
        f"| Images | {b['ai']['images']} | {b['real']['images']} |",
        f"| Square (%) | {b['ai']['square_pct']} | {b['real']['square_pct']} |",
        f"| Stored as PNG (%) | {b['ai']['png_pct']} | {b['real']['png_pct']} |",
        f"| Has EXIF (%) | {b['ai']['exif_pct']} | {b['real']['exif_pct']} |",
        f"| Median megapixels | {b['ai']['median_megapixels']} | {b['real']['median_megapixels']} |",
        f"| Top resolutions | {', '.join(b['ai']['top_resolutions'])} | {', '.join(b['real']['top_resolutions'])} |",
        "",
        "## Shortcut-risk check (no pixels used)",
        "",
        "A logistic-regression model on file metadata only (aspect ratio, resolution, PNG/JPEG, EXIF, file size),",
        "fitted on the train split and evaluated on val and test (same split as the image models):",
        "",
        "| Input | Val accuracy | Val ROC-AUC | Test accuracy | Test ROC-AUC |",
        "|---|---|---|---|---|",
        "| Raw-file metadata | {val_accuracy:.3f} | {val_roc_auc:.3f} | **{test_accuracy:.3f}** | {test_roc_auc:.3f} |".format(**r["metadata_only_baseline_raw"]),
        "| Cleaned-file JPEG size only | {val_accuracy:.3f} | {val_roc_auc:.3f} | **{test_accuracy:.3f}** | {test_roc_auc:.3f} |".format(**r["filesize_only_baseline_cleaned"]),
        "",
        "How to read this: labels in the raw files are highly predictable from non-content cues, so any model",
        "trained on the raw files *could* have used them. This is a risk indicator; it does not show what the",
        "legacy CNNs actually learned. After cleaning, these particular cues carry little signal, but other",
        "shortcuts (e.g. different source resolutions and therefore resampling statistics) may remain.",
        "",
        f"## Split (seed {r['split_seed']}, stratified, file level)",
        "",
        "| Split | AI | Real |",
        "|---|---|---|",
    ]
    for s, c in r["split"].items():
        lines.append(f"| {s} | {c['ai']} | {c['real']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
