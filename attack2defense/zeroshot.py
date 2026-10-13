"""Last week's experiment: ImageNet models on CIFAR-10 cats.

ResNet-50 and ViT-B/16 stay 1000-way ImageNet classifiers. A CIFAR cat image
counts as correct when the predicted ImageNet category is a cat. FGSM and PGD
then try to push that prediction somewhere else.

Category matching uses a whole word, so "catamaran" and "caterpillar" are not
counted as cats.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from attack2defense.attacks import craft_adversarial_examples
from attack2defense.config import IMAGENET_MEAN, IMAGENET_STD, TARGET_EPSILON
from attack2defense.data import make_transforms
from attack2defense.models import load_imagenet_classifier, wrap_model
from attack2defense.plotting import plot_overall
from attack2defense.utils import device_label, get_device, progress


def is_cat_category(name: str) -> bool:
    return re.search(r"\bcat\b", name.lower()) is not None


class _CatSubset(Dataset):
    def __init__(self, base, indices: list[int], transform):
        self.base = base
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, label = self.base[self.indices[index]]
        return self.transform(image), label


def _cat_loader(
    data_dir: str,
    num_images: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    raw = datasets.CIFAR10(root=data_dir, train=False, download=True)
    cat_id = raw.class_to_idx["cat"]
    cat_indices = [index for index, label in enumerate(raw.targets) if int(label) == cat_id]
    if num_images > len(cat_indices):
        raise ValueError(f"CIFAR-10 test has {len(cat_indices)} cats, not {num_images}.")
    subset = _CatSubset(
        raw,
        cat_indices[:num_images],
        make_transforms(image_size=224, train=False),
    )
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def _evaluate_zeroshot(model, loader, cat_ids: set[int], epsilon: float, pgd_steps: int, device) -> dict:
    clean_correct = 0
    fgsm_correct = 0
    pgd_correct = 0
    total = 0
    model.eval()
    for images, _labels in progress(loader, desc="Zero-shot"):
        images = images.to(device)
        total += int(images.shape[0])
        with torch.no_grad():
            clean_pred = model(images).argmax(dim=1)
        clean_correct += int(sum(int(pred) in cat_ids for pred in clean_pred.tolist()))

        fgsm_images = craft_adversarial_examples(model, images, clean_pred, epsilon, kind="fgsm", steps=1)
        with torch.no_grad():
            fgsm_pred = model(fgsm_images).argmax(dim=1)
        fgsm_correct += int(sum(int(pred) in cat_ids for pred in fgsm_pred.tolist()))

        pgd_images = craft_adversarial_examples(
            model, images, clean_pred, epsilon, kind="pgd", steps=pgd_steps
        )
        with torch.no_grad():
            pgd_pred = model(pgd_images).argmax(dim=1)
        pgd_correct += int(sum(int(pred) in cat_ids for pred in pgd_pred.tolist()))

    return {
        "clean_accuracy": clean_correct / total,
        "fgsm_accuracy": fgsm_correct / total,
        "pgd_accuracy": pgd_correct / total,
        "images": total,
        "epsilon": epsilon,
        "pgd_steps": pgd_steps,
    }


def run_zeroshot_benchmark(
    models: list[str],
    output_dir: str | Path,
    *,
    num_images: int = 100,
    epsilon: float = TARGET_EPSILON,
    pgd_steps: int = 10,
    batch_size: int = 8,
    data_dir: str = "data",
    num_workers: int = 0,
) -> pd.DataFrame:
    """Compare ImageNet ResNet-50 and ViT-B/16 on CIFAR-10 cat images."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Zero-shot benchmark on {device_label(device)}", flush=True)
    loader = _cat_loader(data_dir, num_images, batch_size, num_workers, device)

    rows = []
    for name in models:
        backbone, categories = load_imagenet_classifier(name)
        model = wrap_model(backbone, IMAGENET_MEAN, IMAGENET_STD).to(device)
        cat_ids = {index for index, category in enumerate(categories) if is_cat_category(category)}
        cat_names = [categories[index] for index in sorted(cat_ids)]
        print(f"{name} cat categories ({len(cat_names)}): {', '.join(cat_names)}", flush=True)
        metrics = _evaluate_zeroshot(model, loader, cat_ids, epsilon, pgd_steps, device)
        metrics["model"] = name
        metrics["mode"] = "imagenet-zeroshot"
        rows.append(metrics)
        print(
            f"{name}  clean {metrics['clean_accuracy']:.2%}  "
            f"FGSM {metrics['fgsm_accuracy']:.2%}  "
            f"PGD {metrics['pgd_accuracy']:.2%}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    frame.to_csv(output_path / "zeroshot_summary.csv", index=False)
    plot_overall(
        frame,
        output_path / "zeroshot_comparison.png",
        f"ImageNet zero-shot on CIFAR cats  |  epsilon {epsilon:.5f}",
    )
    print(f"Wrote {output_path}", flush=True)
    return frame
