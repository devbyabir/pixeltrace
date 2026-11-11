"""Config loading, seeding and device helpers."""
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
import yaml


@dataclass
class Config:
    # model
    model: str = "resnet50.a1_in1k"     # timm model name, or "simple_cnn"
    pretrained: bool = True
    img_size: int = 224
    drop_rate: float = 0.2
    # data
    data_dir: str = "data/processed"
    num_workers: int = 2
    # optimisation
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 1
    label_smoothing: float = 0.0
    patience: int = 7                    # early stopping on validation ROC-AUC
    amp: bool = True
    # bookkeeping
    seed: int = 42
    run_name: str = ""
    output_dir: str = "results/runs"
    limit: int = 0                       # >0: use only N images per split (smoke tests)
    extra: dict = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / (self.run_name or self.model.split(".")[0])

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def load_config(path: str | os.PathLike | None, overrides: dict | None = None) -> Config:
    data = {}
    if path:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data.update({k: v for k, v in (overrides or {}).items() if v is not None})
    known = {f.name for f in fields(Config)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return Config(**data)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def environment_info() -> dict:
    """Versions and hardware actually used for a run (saved next to every run's metrics)."""
    import platform
    from importlib import metadata

    pkgs = {}
    for name in ("torch", "torchvision", "timm", "numpy", "pandas", "pillow", "scikit-learn",
                 "matplotlib", "pyyaml", "onnx", "onnxruntime"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    info = {"python": platform.python_version(), "platform": platform.platform(), "packages": pkgs,
            "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
