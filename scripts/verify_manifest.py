"""Compare a freshly generated split manifest against the committed reference.

    python scripts/verify_manifest.py --reference results/manifest/splits_manifest.csv \
                                      --candidate /tmp/audit/manifest/splits_manifest.csv

Exit code 1 if ids, split assignment, labels or raw-file hashes differ. Processed-image hash
differences are reported separately (they depend on the Pillow/libjpeg build) and only fail
with --strict-processed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys

CORE = ("id", "split", "label", "source_file", "raw_sha256")


def digest(rows: dict[str, dict], cols) -> str:
    h = hashlib.sha256()
    for i in sorted(rows):
        h.update(("|".join(rows[i][c] for c in cols) + "\n").encode())
    return h.hexdigest()


def load(path: str) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", help="reference splits_manifest.csv")
    ap.add_argument("--reference-digest", help="digest.json (when the full CSV is not available)")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--strict-processed", action="store_true")
    ap.add_argument("--write-digest", help="write digest.json for --candidate and exit")
    a = ap.parse_args()
    cand = load(a.candidate)
    if a.write_digest:
        json.dump({"rows": len(cand), "core_sha256": digest(cand, CORE),
                   "processed_sha256": digest(cand, ("id", "processed_sha256"))}, open(a.write_digest, "w"), indent=2)
        return
    if a.reference_digest:
        ref_d = json.load(open(a.reference_digest))
        core_ok = ref_d["core_sha256"] == digest(cand, CORE) and ref_d["rows"] == len(cand)
        proc_ok = ref_d["processed_sha256"] == digest(cand, ("id", "processed_sha256"))
        print(f"split/label/source/raw-hash digest: {'IDENTICAL' if core_ok else 'DIFFERENT'}")
        print(f"processed-image digest: {'IDENTICAL' if proc_ok else 'DIFFERENT (Pillow/libjpeg build?)'}")
        if not core_ok or (a.strict_processed and not proc_ok):
            sys.exit(1)
        return
    ref = load(a.reference)

    problems = []
    if set(ref) != set(cand):
        problems.append(f"id sets differ: {len(set(ref) ^ set(cand))} ids")
    common = set(ref) & set(cand)
    for key in ("split", "label", "source_file", "raw_sha256"):
        bad = [i for i in common if ref[i][key] != cand[i][key]]
        if bad:
            problems.append(f"{key} differs for {len(bad)} images (e.g. {sorted(bad)[:3]})")
    proc_bad = [i for i in common if ref[i]["processed_sha256"] != cand[i]["processed_sha256"]]

    print(f"reference {len(ref)} rows, candidate {len(cand)} rows")
    print(f"split/label/source/raw-hash: {'IDENTICAL' if not problems else 'DIFFERENT'}")
    print(f"processed-image hashes: {len(common) - len(proc_bad)}/{len(common)} identical")
    for p in problems:
        print("  -", p)
    if problems or (a.strict_processed and proc_bad):
        sys.exit(1)


if __name__ == "__main__":
    main()
