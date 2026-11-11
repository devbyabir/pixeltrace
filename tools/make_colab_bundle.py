"""Pack the code (no data, no results) into a base64 zip that can be pasted into a Colab cell.

    python tools/make_colab_bundle.py   ->  dist/colab_bundle.b64.txt
"""
import base64
import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["src/**/*.py", "scripts/*.py", "configs/**/*.yaml", "pyproject.toml", "requirements.txt",
           "README.md", "DATASET.md", "results/manifest/digest.json"]

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    files = sorted({p for pat in INCLUDE for p in ROOT.glob(pat) if p.is_file() and "__pycache__" not in p.parts})
    for p in files:
        z.write(p, p.relative_to(ROOT).as_posix())
(ROOT / "dist").mkdir(exist_ok=True)
(ROOT / "dist" / "pixeltrace_code.zip").write_bytes(buf.getvalue())
out = ROOT / "dist" / "colab_bundle.b64.txt"
out.parent.mkdir(exist_ok=True)
out.write_text(base64.b64encode(buf.getvalue()).decode(), encoding="ascii")
print(f"{len(files)} files, zip {len(buf.getvalue())} B, base64 {out.stat().st_size} B -> {out}")
