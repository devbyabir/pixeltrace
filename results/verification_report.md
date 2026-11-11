# PixelTrace v1.0.0: verification report

Date: 2026-11-12 · Training: Google Colab, Tesla T4, PyTorch 2.14.1+cu130, timm 1.0.31 (exact versions: `requirements-lock.txt`)

## Results (300-image test split, evaluated once; checkpoints and deployment model chosen on validation)

| Model | Val ROC-AUC | Test accuracy (95 % CI) | Test ROC-AUC (95 % CI) |
|---|---|---|---|
| EfficientNet-B0 (deployed) | 0.952 | 0.843 (0.803–0.887) | 0.904 (0.866–0.940) |
| ResNet-50 | 0.935 | 0.807 (0.763–0.850) | 0.897 (0.859–0.931) |
| Simple CNN | 0.877 | 0.767 (0.720–0.813) | 0.835 (0.788–0.881) |

Two independent full runs produced identical test metrics, and an ONNX file with the same SHA-256.

## Checks

| Check | Where | Outcome |
|---|---|---|
| Smoke run of the whole pipeline (1 epoch, 64 images/split) | Colab CPU | PASSED |
| Rebuilt split vs committed manifest (ids, split, labels, raw SHA-256, processed SHA-256) | Colab; fresh clone on Windows | identical (2000/2000) |
| Checkpoint reload reproduces saved test predictions | Colab, `pixeltrace.verify` | max \|ΔP\| ≤ 4.3e-6 |
| Raw original file → same score as cleaned copy | Colab, 20 images/model | max \|ΔP\| ≤ 1.8e-6 |
| `metrics.json` = metrics recomputed from per-image predictions; ids/labels = manifest | Windows, `tools/check_results.py` (no PyTorch) | 25/25 PASS |
| Deployment model = best validation ROC-AUC | both | PASS |
| PyTorch vs ONNX Runtime (300 test images) | Colab | max \|ΔP\| = 5.5e-6 |
| JS port of Pillow resampler vs Pillow (7 samples + 60 random sizes) | Node | bit-exact |
| Browser preprocessing vs Python (decode, EXIF, alpha, resize, JPEG q95, 224 input) | Chromium | bit-exact on 7/7 samples |
| Browser ONNX Runtime Web vs Python ONNX Runtime | Chromium | max \|ΔP\| = 2.7e-5, labels agree |
| Browser on 8 original dataset files vs training-time predictions | Chromium (private, temporary copies, deleted) | max \|ΔP\| = 4.5e-6, labels agree |
| Demo states: loading, result, decode error; no horizontal scroll at 375 px | Chromium | PASS |
| Backup safeguards: full mode refuses without backup; write/read probe; best checkpoint mirrored on improvement; size-checked copies | local unit check + Colab log | PASS |
| Downloaded result ZIPs readable (`testzip`) | Windows | simple_cnn, final: OK |
| Public package: no dataset images, checkpoints, private dataset link, account data, credentials, local paths | git file list audit | clean (112 files, 21.8 MB) |

## Deployed website (https://devbyabir.github.io/pixeltrace/)

| Check | Outcome |
|---|---|
| Every deployed file (pages, scripts, results.json, ONNX model) byte-identical to the repository | MATCH (model SHA-256 `710f9b71…`) |
| Live parity page in Chrome 157 (7 synthetic samples) | preprocessing bit-exact; max \|ΔP\| = 2.7e-5; labels agree |
| Live demo, 8 real test images uploaded through the page's file picker (`live_demo_check.json`) | max \|ΔP\| vs verified Python predictions = 3.3e-6; 8/8 labels agree, including the 3 images the model gets wrong |
| Network during uploads | only GET requests for ONNX Runtime Web (jsDelivr) and the model; no image leaves the browser |
| Dashboard and report rendered from results.json; no horizontal scroll at 375 px | PASS |

## Not covered

- Training was not re-run on this Windows PC, where PyTorch DLLs are blocked by Windows Application Control. Every PyTorch step ran on fresh Colab VMs.
- The site was tested in Chromium-based browsers only (Chrome 157 and the Claude desktop browser); not in Firefox or Safari.
- Generalisation to other generators or datasets was not tested (see README limitations).
