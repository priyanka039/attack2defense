"""CIFAR-10 splits for training, validation, and the official test set."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from attack2defense.config import CIFAR10_CLASSES


def make_transforms(image_size: int, train: bool) -> transforms.Compose:
    """Augment the training split. Evaluation images are only resized."""

    operations: list = []
    if train:
        operations.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    if image_size != 32:
        operations.append(transforms.Resize((image_size, image_size)))
    operations.append(transforms.ToTensor())
    return transforms.Compose(operations)


def stratified_indices(
    targets: list[int] | tuple[int, ...],
    val_per_class: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Hold out the same number of images from every class."""

    by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(targets):
        by_class[int(label)].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for label in sorted(by_class):
        indices = by_class[label][:]
        rng.shuffle(indices)
        if val_per_class <= 0 or val_per_class >= len(indices):
            raise ValueError(
                f"val_per_class must leave at least one training image; "
                f"class {label} has {len(indices)} images and "
                f"val_per_class={val_per_class}."
            )
        val_indices.extend(indices[:val_per_class])
        train_indices.extend(indices[val_per_class:])
    return train_indices, val_indices


def sample_indices_per_class(
    targets: list[int] | tuple[int, ...],
    per_class: int,
    seed: int,
) -> list[int]:
    """Take up to ``per_class`` images from each class, in a fixed order."""

    by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(targets):
        by_class[int(label)].append(index)

    rng = random.Random(seed)
    chosen: list[int] = []
    for label in sorted(by_class):
        indices = by_class[label][:]
        rng.shuffle(indices)
        chosen.extend(indices[:per_class])
    return chosen


def first_index_per_class(targets: list[int] | tuple[int, ...], num_classes: int) -> list[int]:
    """First occurrence of each class, so example grids stay reproducible."""

    found: dict[int, int] = {}
    for index, label in enumerate(targets):
        label = int(label)
        if label not in found:
            found[label] = index
        if len(found) == num_classes:
            break
    missing = [label for label in range(num_classes) if label not in found]
    if missing:
        raise RuntimeError(f"Dataset is missing classes: {missing}")
    return [found[label] for label in range(num_classes)]


def _loader_kwargs(num_workers: int, device: torch.device) -> dict:
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


@dataclass
class CIFARData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    test_dataset: Dataset
    class_names: tuple[str, ...]


def load_cifar10(
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_per_class: int,
    device: torch.device,
) -> CIFARData:
    """Download CIFAR-10 and build train, validation, and test loaders.

    Validation images come from the training split, with augmentations turned
    off. The official test split is reserved for the final report.
    """

    train_transform = make_transforms(image_size, train=True)
    eval_transform = make_transforms(image_size, train=False)

    training_source = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    validation_source = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=eval_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )

    train_indices, val_indices = stratified_indices(
        training_source.targets,
        val_per_class=val_per_class,
        seed=seed,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    shared = _loader_kwargs(num_workers, device)

    train_loader = DataLoader(
        Subset(training_source, train_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        **shared,
    )
    val_loader = DataLoader(
        Subset(validation_source, val_indices),
        batch_size=batch_size,
        shuffle=False,
        **shared,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **shared,
    )
    return CIFARData(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_dataset=test_dataset,
        class_names=CIFAR10_CLASSES,
    )


def make_test_loader(
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_per_class: int | None = None,
    seed: int = 42,
) -> tuple[DataLoader, Dataset]:
    """Official CIFAR-10 test loader, optionally capped per class."""

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=make_transforms(image_size, train=False),
    )
    if max_per_class is not None:
        indices = sample_indices_per_class(dataset.targets, max_per_class, seed)
        dataset_for_loader: Dataset = Subset(dataset, indices)
    else:
        dataset_for_loader = dataset
    loader = DataLoader(
        dataset_for_loader,
        batch_size=batch_size,
        shuffle=False,
        **_loader_kwargs(num_workers, device),
    )
    return loader, dataset
