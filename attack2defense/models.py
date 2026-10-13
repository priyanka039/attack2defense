"""Classifiers used by the training pipeline and the zero-shot benchmark."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet50_Weights,
    ViT_B_16_Weights,
    resnet18,
    resnet50,
    vit_b_16,
)

from attack2defense.config import MODEL_SPECS, ModelSpec


class PixelModel(nn.Module):
    """Backbone that accepts images in [0, 1] and normalizes internally.

    Putting normalization inside the module lets FGSM and PGD edit pixels
    directly. An epsilon of 2/255 is then two intensity levels, on every
    channel, rather than a shift in normalized coordinates.
    """

    def __init__(
        self,
        backbone: nn.Module,
        mean: tuple[float, float, float] | list[float],
        std: tuple[float, float, float] | list[float],
    ):
        super().__init__()
        self.backbone = backbone
        mean_tensor = torch.tensor(list(mean), dtype=torch.float32).view(1, 3, 1, 1)
        std_tensor = torch.tensor(list(std), dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", std_tensor)
        self.channels_last = False

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if self.channels_last:
            pixels = pixels.contiguous(memory_format=torch.channels_last)
        normalized = (pixels - self.mean) / self.std
        return self.backbone(normalized)


class SmallCIFARCNN(nn.Module):
    """CIFAR-10 network sized so one adversarial epoch fits on a laptop CPU.

    Five convolutions, batch norm, and global average pooling. A 256-wide
    fully connected layer on 4x4 maps was the expensive part of the previous
    version; pooling removes it. On this machine that is the difference
    between a 30 minute epoch and a 100 minute ResNet-18 epoch.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 32),
            nn.MaxPool2d(kernel_size=2),
            self._block(32, 64),
            self._block(64, 64),
            nn.MaxPool2d(kernel_size=2),
            self._block(64, 128),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(pixels))


def _replace_resnet_head(model: nn.Module, num_classes: int) -> None:
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)


def _replace_vit_head(model: nn.Module, num_classes: int) -> None:
    if not hasattr(model, "heads") or not hasattr(model.heads, "head"):
        raise TypeError("This torchvision ViT build has no heads.head layer.")
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)


def build_backbone(
    name: str,
    num_classes: int = 10,
    pretrained: bool | None = None,
) -> nn.Module:
    """Create a raw backbone. Normalization is added by ``PixelModel``."""

    if name not in MODEL_SPECS:
        known = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model '{name}'. Choose from: {known}")

    spec = MODEL_SPECS[name]
    use_pretrained = spec.pretrained if pretrained is None else pretrained

    if name == "resnet18":
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        _replace_resnet_head(model, num_classes)
        return model

    if name == "small_cnn":
        return SmallCIFARCNN(num_classes=num_classes)

    if name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if use_pretrained else None
        model = resnet50(weights=weights)
        _replace_resnet_head(model, num_classes)
        return model

    if name == "vit_b_16":
        weights = ViT_B_16_Weights.DEFAULT if use_pretrained else None
        model = vit_b_16(weights=weights)
        _replace_vit_head(model, num_classes)
        return model

    raise ValueError(f"Unknown model '{name}'")


def wrap_model(
    backbone: nn.Module,
    mean: tuple[float, float, float] | list[float],
    std: tuple[float, float, float] | list[float],
) -> PixelModel:
    return PixelModel(backbone, mean, std)


def build_pixel_model(
    name: str,
    num_classes: int = 10,
    pretrained: bool | None = None,
) -> PixelModel:
    spec: ModelSpec = MODEL_SPECS[name]
    backbone = build_backbone(name, num_classes=num_classes, pretrained=pretrained)
    return wrap_model(backbone, spec.mean, spec.std)


def load_imagenet_classifier(name: str) -> tuple[nn.Module, list[str]]:
    """ImageNet classifier with its category list, for the zero-shot benchmark."""

    if name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    elif name == "vit_b_16":
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)
    else:
        raise ValueError(f"The zero-shot benchmark supports resnet50 and vit_b_16, not '{name}'.")
    categories = [str(category) for category in weights.meta["categories"]]
    return model, categories


CONV_MODELS = {"small_cnn", "resnet18", "resnet50"}


def use_channels_last(model: PixelModel) -> PixelModel:
    """Store convolutions in channels-last memory. This is faster on CPU."""

    model.to(memory_format=torch.channels_last)
    model.channels_last = True
    return model


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def optimizer_groups(
    backbone: nn.Module,
    name: str,
    learning_rate: float,
    weight_decay: float,
) -> list[dict]:
    """Use a smaller learning rate on a pretrained trunk than on the new head."""

    head: list[nn.Parameter]
    if name == "resnet50":
        head = list(backbone.fc.parameters())
    elif name == "vit_b_16":
        head = list(backbone.heads.parameters())
    else:
        head = []

    if not head:
        return [
            {
                "params": [parameter for parameter in backbone.parameters() if parameter.requires_grad],
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        ]

    head_ids = {id(parameter) for parameter in head}
    trunk = [
        parameter
        for parameter in backbone.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    return [
        {"params": trunk, "lr": learning_rate * 0.1, "weight_decay": weight_decay},
        {"params": head, "lr": learning_rate, "weight_decay": weight_decay},
    ]
