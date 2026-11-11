# Dataset card

## Provenance (what is known, and what is not)

- The dataset (`ai_vs_real_img_dataset`, a ~3.6 GB zip on Google Drive) was **received from a senior colleague** who had previously worked on this topic. The author of PixelTrace used it for these experiments.
- The **original source, the image generators and the redistribution licence are unknown.** Neither the author nor the colleague is known to have created or to own the images.
- Because of that, this repository and its website contain **no dataset images and no image-containing outputs** derived from them (Grad-CAM overlays, error grids, sample grids). Only numbers, plots of numbers, file names, hashes and per-image scores are published. This stays the case until permission is confirmed.
- If you know the original source of these images, please open an issue so it can be credited.

| Field | Value |
|---|---|
| Name | ai_vs_real_img_dataset (v001) |
| Obtained from | A senior colleague (personal communication) |
| Download | Not public. The repository does not contain or link to the images |
| Original source | Unknown |
| AI generators | Unknown (resolutions such as 1024×1024, 1456×816 and 2048×2048 are typical of common text-to-image tools; unverified) |
| Real photos | Unknown (resolutions such as 1080×1620 and 6000×4000 are typical of stock or camera photos; unverified) |
| Licence | Unknown, so not redistributed |

## Raw contents (as received)

- 3,998 files in two folders, `AI Generated/` (1,999) and `Real/` (1,999)
- 2,000 distinct images: **1,998 occur twice** under different file names, **2 occur once**
- 726 AI images are PNG files with a `.jpg` extension

Full numbers: [`results/data_audit.md`](results/data_audit.md).

## Reproducing the split without the images in this repo

`results/manifest/splits_manifest.csv` lists, for each of the 2,000 images, its id, split, label, original file name, and the SHA-256 of both the raw file and the cleaned file. `results/manifest/digest.json` summarises it. With the same raw files:

```bash
python scripts/prepare_data.py --raw data/raw --out data/processed --report-dir /tmp/audit
python scripts/verify_manifest.py --reference results/manifest/splits_manifest.csv \
                                  --candidate /tmp/audit/manifest/splits_manifest.csv
```

Raw hashes, split and labels must match exactly. Cleaned-file hashes also match when the same Pillow build is used (13.0.0 here); other builds may re-encode JPEGs slightly differently.

## Getting the data

The images are not distributed with this repository, and no download link is published. If you have been given
access, either place the folders in `data/raw/` or set `PIXELTRACE_RAW_GDRIVE_ID` to the Google Drive file id of
the zip before running `scripts/colab_pipeline.py`.

## Expected local layout

```
data/raw/AI Generated/*.jpg
data/raw/Real/*.jpg
```
