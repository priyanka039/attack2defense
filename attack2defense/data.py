"""CIFAR-10 splits for training, validation, and the official test set."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from attack2defense.config import CIFAR10_CLASSES, DATASET_NAMES, IMAGENETTE_CLASSES


def make_transforms(image_size: int, train: bool, dataset: str = "cifar10") -> transforms.Compose:
    """Augment the training split. Evaluation images are only resized."""

    operations: list = []
    if dataset == "imagenette":
        if train:
            operations.extend(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            operations.append(transforms.Resize((image_size, image_size)))
        operations.append(transforms.ToTensor())
        return transforms.Compose(operations)

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
    dataset_name: str = "cifar10"


def dataset_classes(name: str) -> tuple[str, ...]:
    if name == "cifar10":
        return CIFAR10_CLASSES
    if name == "imagenette":
        return IMAGENETTE_CLASSES
    raise ValueError(f"Unknown dataset '{name}'. Choose from: {', '.join(DATASET_NAMES)}")


def default_image_size(dataset: str, model_image_size: int) -> int:
    """Imagenette is trained at 128px; CIFAR models keep their native size."""

    if dataset == "imagenette" and model_image_size <= 32:
        return 128
    return model_image_size


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

    train_transform = make_transforms(image_size, train=True, dataset="cifar10")
    eval_transform = make_transforms(image_size, train=False, dataset="cifar10")

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
        dataset_name="cifar10",
    )


def _imagenette(data_dir: str, split: str, transform) -> Dataset:
    return datasets.Imagenette(
        root=data_dir,
        split=split,
        size="160px",
        download=True,
        transform=transform,
    )


def load_imagenette(
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_per_class: int,
    device: torch.device,
) -> CIFARData:
    """Imagenette: 10 ImageNet classes, 160px source images resized to ``image_size``.

    The official validation split is the held-out test set. A seeded slice of
    the training split is used for validation, matching the CIFAR-10 protocol.
    """

    train_transform = make_transforms(image_size, train=True, dataset="imagenette")
    eval_transform = make_transforms(image_size, train=False, dataset="imagenette")
    training_source = _imagenette(data_dir, "train", train_transform)
    validation_source = _imagenette(data_dir, "train", eval_transform)
    test_dataset = _imagenette(data_dir, "val", eval_transform)
    train_indices, val_indices = stratified_indices(
        training_source.targets, val_per_class=val_per_class, seed=seed
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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **shared)
    return CIFARData(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_dataset=test_dataset,
        class_names=IMAGENETTE_CLASSES,
        dataset_name="imagenette",
    )


def load_dataset(
    name: str,
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_per_class: int,
    device: torch.device,
) -> CIFARData:
    """Dispatch to CIFAR-10 or Imagenette with the same seeded split protocol."""

    dataset_classes(name)
    loader = load_cifar10 if name == "cifar10" else load_imagenette
    return loader(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        val_per_class=val_per_class,
        device=device,
    )


def make_test_loader(
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_per_class: int | None = None,
    seed: int = 42,
    dataset: str = "cifar10",
) -> tuple[DataLoader, Dataset]:
    """Held-out test loader, optionally capped per class."""

    dataset_classes(dataset)
    transform = make_transforms(image_size, train=False, dataset=dataset)
    if dataset == "imagenette":
        test_set = _imagenette(data_dir, "val", transform)
    else:
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    if max_per_class is not None:
        indices = sample_indices_per_class(test_set.targets, max_per_class, seed)
        dataset_for_loader: Dataset = Subset(test_set, indices)
    else:
        dataset_for_loader = test_set
    loader = DataLoader(
        dataset_for_loader,
        batch_size=batch_size,
        shuffle=False,
        **_loader_kwargs(num_workers, device),
    )
    return loader, test_set
