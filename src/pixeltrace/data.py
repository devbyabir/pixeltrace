"""Dataset + transforms driven by data/processed/splits.csv (written by scripts/prepare_data.py)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


class SplitDataset(Dataset):
    """Images of one split. `corrupt` (PIL -> PIL) is applied before the transform (robustness tests)."""

    def __init__(self, data_dir: str | Path, split: str, transform: Callable,
                 corrupt: Callable[[Image.Image], Image.Image] | None = None, limit: int = 0):
        self.data_dir = Path(data_dir)
        df = pd.read_csv(self.data_dir / "splits.csv")
        df = df[df["split"] == split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows for split '{split}' in {self.data_dir / 'splits.csv'}")
        if limit:
            # keep the class balance when sub-sampling for smoke tests
            df = pd.concat([g.sample(min(len(g), max(1, limit // 2)), random_state=0)
                            for _, g in df.groupby("label")]).reset_index(drop=True)
        self.df = df
        self.transform = transform
        self.corrupt = corrupt

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(self.data_dir / row["path"]).convert("RGB")
        if self.corrupt is not None:
            img = self.corrupt(img)
        return self.transform(img), int(row["label"]), i


def build_transforms(img_size: int, mean, std, train: bool) -> Callable:
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    if train:
        return T.Compose([
            # mild geometric augmentation only: colour/blur augmentation would also erase the
            # low-level generator artifacts the detector is supposed to learn
            T.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            *norm,
        ])
    return T.Compose([T.Resize(img_size), T.CenterCrop(img_size), *norm])


def build_loader(data_dir, split: str, img_size: int, mean, std, batch_size: int,
                 num_workers: int, train: bool = False, corrupt=None, limit: int = 0) -> DataLoader:
    ds = SplitDataset(data_dir, split, build_transforms(img_size, mean, std, train), corrupt, limit)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=train and len(ds) > batch_size,
                      persistent_workers=num_workers > 0)
