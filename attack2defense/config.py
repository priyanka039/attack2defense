"""Shared constants for data, models, and the attack budget."""

from __future__ import annotations

from dataclasses import dataclass

CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Ten ImageNet classes used by Imagenette (https://github.com/fastai/imagenette).
IMAGENETTE_CLASSES: tuple[str, ...] = (
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
)

DATASET_NAMES = ("cifar10", "imagenette")

# Statistics used by the usual CIFAR-10 training recipe.
CIFAR_MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR_STD: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

# Statistics baked into torchvision's ImageNet weights.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# L-infinity ball in pixel space. Images live in [0, 1], so 2/255 means
# each channel may move by at most two intensity levels out of 255.
TARGET_EPSILON: float = 2 / 255
ROBUST_ACCURACY_TARGET: float = 0.80

# The budget used in most CIFAR-10 robustness papers. A single model trained
# from scratch rarely stays above 80% under this larger attack.
STRONG_EPSILON: float = 8 / 255


@dataclass(frozen=True)
class ModelSpec:
    """How a named architecture should see CIFAR-10 images."""

    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    pretrained: bool
    description: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "resnet18": ModelSpec(
        image_size=32,
        mean=CIFAR_MEAN,
        std=CIFAR_STD,
        pretrained=False,
        description=(
            "ResNet-18 adapted to 32x32 CIFAR images (3x3 stride-1 stem, "
            "no initial max-pool). This is the practical CPU default."
        ),
    ),
    "small_cnn": ModelSpec(
        image_size=32,
        mean=CIFAR_MEAN,
        std=CIFAR_STD,
        pretrained=False,
        description=(
            "Small CIFAR-10 CNN with global average pooling. "
            "On the order of 15-20 minutes per adversarial epoch on a laptop CPU."
        ),
    ),
    "resnet50": ModelSpec(
        image_size=224,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        pretrained=True,
        description=("ImageNet ResNet-50 with a new 10-way classifier. Fine-tuning it wants a GPU."),
    ),
    "vit_b_16": ModelSpec(
        image_size=224,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        pretrained=True,
        description=(
            "ImageNet ViT-B/16 (16x16 patches, 224x224 input) with a new "
            "10-way classifier. Fine-tuning it wants a GPU."
        ),
    ),
}
