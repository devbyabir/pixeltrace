"""Model factory: a from-scratch baseline CNN plus any timm backbone."""
from __future__ import annotations

import timm
import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _block(c_in: int, c_out: int, n_convs: int, drop: float) -> nn.Sequential:
    layers = []
    for k in range(n_convs):
        layers += [nn.Conv2d(c_in if k == 0 else c_out, c_out, 3, padding=1, bias=False),
                   nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(2), nn.Dropout(drop)]
    return nn.Sequential(*layers)


class SimpleCNN(nn.Module):
    """VGG-style baseline trained from scratch (cleaned-up version of the original custom CNN)."""

    def __init__(self, num_classes: int = 2, drop_rate: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            _block(3, 32, 2, drop_rate),
            _block(32, 64, 2, drop_rate),
            _block(64, 128, 3, drop_rate),
            _block(128, 256, 3, drop_rate),
            _block(256, 256, 2, drop_rate),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop_rate),
                                  nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(name: str, pretrained: bool = True, num_classes: int = 2, drop_rate: float = 0.2) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNN(num_classes, drop_rate)
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)


def normalization(model: nn.Module) -> tuple[tuple, tuple]:
    """Mean/std the backbone was pre-trained with (fixes the missing preprocessing in the old notebooks)."""
    if isinstance(model, SimpleCNN):
        return (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    cfg = timm.data.resolve_data_config({}, model=model)
    return tuple(cfg.get("mean", IMAGENET_MEAN)), tuple(cfg.get("std", IMAGENET_STD))


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Last spatial feature layer for Grad-CAM."""
    if isinstance(model, SimpleCNN):
        return model.features[-1][-3]  # last ReLU of the last block (before pooling)
    if hasattr(model, "blocks") and hasattr(model, "patch_embed"):  # ViT / DeiT style
        return model.blocks[-1].norm1
    # order matters: e.g. MobileNetV3's conv_head runs *after* global pooling (1x1 map)
    for attr in ("layer4", "blocks", "stages", "features", "conv_head"):
        if hasattr(model, attr):
            m = getattr(model, attr)
            return m[-1] if isinstance(m, (nn.Sequential, nn.ModuleList)) else m
    raise ValueError(f"Don't know which layer to use for Grad-CAM on {type(model).__name__}")
