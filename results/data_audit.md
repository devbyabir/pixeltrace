# Data audit

- Raw files: **3998**
- Exact duplicate copies: **1998** (2 distinct images occur 1x; 1998 distinct images occur 2x)
- Unique images kept after exact + near-duplicate removal (dHash ≤ 4): **2000**
- Cross-class duplicate groups dropped: 0

## Format / shape bias in the raw files (before de-duplication)

| | AI | Real |
|---|---|---|
| Images | 1999 | 1999 |
| Square (%) | 62.1 | 0.8 |
| Stored as PNG (%) | 36.3 | 0.1 |
| Has EXIF (%) | 51.8 | 9.5 |
| Median megapixels | 1.05 | 1.75 |
| Top resolutions | 1024x1024: 934, 1456x816: 134, 2048x2048: 134 | 1080x1620: 300, 1080x720: 132, 1080x1350: 64 |

## Shortcut-risk check (no pixels used)

A logistic-regression model on file metadata only (aspect ratio, resolution, PNG/JPEG, EXIF, file size),
fitted on the train split and evaluated on val and test (same split as the image models):

| Input | Val accuracy | Val ROC-AUC | Test accuracy | Test ROC-AUC |
|---|---|---|---|---|
| Raw-file metadata | 0.977 | 0.990 | **0.977** | 0.982 |
| Cleaned-file JPEG size only | 0.560 | 0.565 | **0.543** | 0.564 |

How to read this: labels in the raw files are highly predictable from non-content cues, so any model
trained on the raw files *could* have used them. This is a risk indicator; it does not show what the
legacy CNNs actually learned. After cleaning, these particular cues carry little signal, but other
shortcuts (e.g. different source resolutions and therefore resampling statistics) may remain.

## Split (seed 42, stratified, file level)

| Split | AI | Real |
|---|---|---|
| train | 700 | 700 |
| val | 150 | 150 |
| test | 150 | 150 |
