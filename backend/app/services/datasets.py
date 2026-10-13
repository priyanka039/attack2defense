"""Held-out evaluation images served to the attack lab and evaluation jobs."""

from __future__ import annotations

import threading

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from attack2defense.config import CIFAR10_CLASSES, IMAGENETTE_CLASSES
from attack2defense.data import sample_indices_per_class
from backend.app.errors import ApiError
from backend.app.settings import Settings

DATASETS = {
    "cifar10": {
        "name": "cifar10",
        "title": "CIFAR-10",
        "classes": list(CIFAR10_CLASSES),
        "image_size": 32,
        "splits": {"train": 49000, "validation": 1000, "test": 10000},
        "description": "60,000 32x32 colour images in 10 classes. The official test split is held out.",
    },
    "imagenette": {
        "name": "imagenette",
        "title": "Imagenette",
        "classes": list(IMAGENETTE_CLASSES),
        "image_size": 128,
        "splits": {"train": 8469, "validation": 1000, "test": 3925},
        "description": "Ten ImageNet classes at 160px, resized to 128px.",
    },
}


class DatasetService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._cache: dict[str, Dataset] = {}
        self._lock = threading.Lock()

    def info(self, name: str) -> dict:
        if name not in DATASETS:
            known = ", ".join(DATASETS)
            raise ApiError(404, "dataset_not_found", f"Unknown dataset '{name}'. Available: {known}.")
        return DATASETS[name]

    def list(self) -> list[dict]:
        return [self.info(name) for name in DATASETS]

    def test_set(self, name: str) -> Dataset:
        """The test split as tensors in [0, 1]. ``targets`` holds the labels."""

        self.info(name)
        with self._lock:
            if name not in self._cache:
                self._cache[name] = self._load(name)
            return self._cache[name]

    def _load(self, name: str) -> Dataset:
        try:
            if name == "imagenette":
                return datasets.Imagenette(
                    root=str(self.settings.data_dir),
                    split="val",
                    size="160px",
                    download=self.settings.download_data,
                    transform=transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()]),
                )
            return datasets.CIFAR10(
                root=str(self.settings.data_dir),
                train=False,
                download=self.settings.download_data,
                transform=transforms.ToTensor(),
            )
        except RuntimeError as exc:
            raise ApiError(
                503,
                "dataset_unavailable",
                f"{name} is not downloaded. Run a training command once, or set A2D_DOWNLOAD_DATA=true.",
            ) from exc

    def class_index(self, name: str, label: str) -> int:
        classes = self.info(name)["classes"]
        if label not in classes:
            raise ApiError(422, "invalid_label", f"'{label}' is not a {name} class.", "label")
        return classes.index(label)

    def items(self, name: str, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        dataset = self.test_set(name)
        for index in indices:
            if not 0 <= index < len(dataset):
                raise ApiError(
                    422,
                    "invalid_sample",
                    f"Sample index {index} is outside the test set (0 to {len(dataset) - 1}).",
                    "sample_indices",
                )
        images = torch.stack([dataset[index][0] for index in indices])
        labels = torch.tensor([int(dataset.targets[index]) for index in indices])
        return images, labels

    def samples(self, name: str, label: str | None, offset: int, limit: int) -> tuple[list[int], int]:
        """Test-set indices, optionally of one class, plus the total that match."""

        targets = list(self.test_set(name).targets)
        if label is None:
            matching = list(range(len(targets)))
        else:
            wanted = self.class_index(name, label)
            matching = [index for index, target in enumerate(targets) if int(target) == wanted]
        return matching[offset : offset + limit], len(matching)

    def balanced_loader(self, name: str, per_class: int, batch_size: int, seed: int = 42) -> DataLoader:
        """The same class-balanced test subset for every model, so results compare."""

        dataset = self.test_set(name)
        indices = sample_indices_per_class(list(dataset.targets), per_class, seed)
        return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
