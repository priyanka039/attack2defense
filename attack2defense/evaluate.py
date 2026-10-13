"""Measure clean, FGSM, and PGD accuracy on every CIFAR-10 class."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from attack2defense.attacks import craft_adversarial_examples
from attack2defense.config import CIFAR10_CLASSES, ROBUST_ACCURACY_TARGET
from attack2defense.data import first_index_per_class, make_test_loader
from attack2defense.models import (
    CONV_MODELS,
    build_backbone,
    parameter_count,
    use_channels_last,
    wrap_model,
)
from attack2defense.plotting import plot_attack_examples, plot_overall, plot_per_class
from attack2defense.utils import device_label, get_device, load_checkpoint, progress


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _count_matches(labels: torch.Tensor, predictions: torch.Tensor, num_classes: int) -> torch.Tensor:
    labels_cpu = labels.detach().cpu()
    matches = predictions.detach().cpu().eq(labels_cpu)
    return torch.bincount(labels_cpu[matches], minlength=num_classes)


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    epsilon: float,
    pgd_steps: int,
    device: torch.device,
    class_names: tuple[str, ...] | list[str] = CIFAR10_CLASSES,
    max_batches: int | None = None,
    desc: str = "Evaluating",
) -> dict:
    """Run clean, FGSM, and PGD evaluation. Attacks use the true labels."""

    num_classes = len(class_names)
    totals = torch.zeros(num_classes, dtype=torch.long)
    clean_correct = torch.zeros(num_classes, dtype=torch.long)
    fgsm_correct = torch.zeros(num_classes, dtype=torch.long)
    pgd_correct = torch.zeros(num_classes, dtype=torch.long)

    was_training = model.training
    model.eval()
    bar = progress(loader, desc=desc)
    for batch_index, (images, labels) in enumerate(bar):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        totals += torch.bincount(labels.detach().cpu(), minlength=num_classes)

        with torch.no_grad():
            clean_pred = model(images).argmax(dim=1)
        clean_correct += _count_matches(labels, clean_pred, num_classes)

        fgsm_images = craft_adversarial_examples(model, images, labels, epsilon, kind="fgsm", steps=1)
        with torch.no_grad():
            fgsm_pred = model(fgsm_images).argmax(dim=1)
        fgsm_correct += _count_matches(labels, fgsm_pred, num_classes)

        pgd_images = craft_adversarial_examples(model, images, labels, epsilon, kind="pgd", steps=pgd_steps)
        with torch.no_grad():
            pgd_pred = model(pgd_images).argmax(dim=1)
        pgd_correct += _count_matches(labels, pgd_pred, num_classes)

        seen = int(totals.sum())
        bar.set_postfix(clean=f"{_safe_div(int(clean_correct.sum()), seen):.1%}")

    if was_training:
        model.train()

    seen = int(totals.sum())
    if seen == 0:
        raise RuntimeError("Evaluation saw no images.")

    per_class = []
    for index, name in enumerate(class_names):
        total = int(totals[index])
        per_class.append(
            {
                "class": name,
                "images": total,
                "clean_accuracy": _safe_div(int(clean_correct[index]), total),
                "fgsm_accuracy": _safe_div(int(fgsm_correct[index]), total),
                "pgd_accuracy": _safe_div(int(pgd_correct[index]), total),
            }
        )

    return {
        "images": seen,
        "epsilon": epsilon,
        "pgd_steps": pgd_steps,
        "clean_accuracy": _safe_div(int(clean_correct.sum()), seen),
        "fgsm_accuracy": _safe_div(int(fgsm_correct.sum()), seen),
        "pgd_accuracy": _safe_div(int(pgd_correct.sum()), seen),
        "per_class": per_class,
    }


def restore_model(checkpoint_path: str | Path, device: torch.device | None = None):
    """Rebuild the pixel-space model stored in a training checkpoint."""

    if device is None:
        device = get_device()
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]
    backbone = build_backbone(
        config["model"],
        num_classes=int(config["num_classes"]),
        pretrained=False,
    )
    backbone.load_state_dict(checkpoint["backbone"])
    model = wrap_model(backbone, config["mean"], config["std"])
    model.to(device)
    if config["model"] in CONV_MODELS:
        use_channels_last(model)
    model.eval()
    return model, checkpoint, device


def save_example_grid(
    model: torch.nn.Module,
    dataset,
    class_names: list[str],
    device: torch.device,
    epsilon: float,
    pgd_steps: int,
    path: Path,
) -> None:
    """Attack the first test image of each class, then save the grid and its predictions.

    Every model is shown the same ten images, so grids from different models
    can be compared side by side.
    """

    targets = list(dataset.targets)
    indices = first_index_per_class(targets, len(class_names))
    images = torch.stack([dataset[index][0] for index in indices]).to(device)
    labels = torch.arange(len(class_names), device=device)

    model.eval()
    fgsm_images = craft_adversarial_examples(model, images, labels, epsilon, kind="fgsm")
    pgd_images = craft_adversarial_examples(model, images, labels, epsilon, kind="pgd", steps=pgd_steps)
    with torch.no_grad():
        clean_pred = model(images).argmax(dim=1).tolist()
        fgsm_pred = model(fgsm_images).argmax(dim=1).tolist()
        pgd_pred = model(pgd_images).argmax(dim=1).tolist()

    plot_attack_examples(
        images.cpu(),
        fgsm_images.cpu(),
        pgd_images.cpu(),
        labels.cpu().tolist(),
        clean_pred,
        fgsm_pred,
        pgd_pred,
        class_names,
        path,
        epsilon,
    )
    pd.DataFrame(
        {
            "dataset_index": indices,
            "true_class": [class_names[label] for label in labels.cpu().tolist()],
            "clean_prediction": [class_names[label] for label in clean_pred],
            "fgsm_prediction": [class_names[label] for label in fgsm_pred],
            "pgd_prediction": [class_names[label] for label in pgd_pred],
        }
    ).to_csv(path.with_suffix(".csv"), index=False)


def write_evaluation_files(
    result: dict,
    output_dir: Path,
    *,
    model_name: str,
    mode: str,
    class_names: list[str],
    save_examples_from: tuple | None = None,
    threshold: float = ROBUST_ACCURACY_TARGET,
) -> dict:
    """Write CSV, JSON, and plots for one evaluation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    target_met = bool(result["pgd_accuracy"] >= threshold)
    summary = {
        "model": model_name,
        "mode": mode,
        "epsilon": result["epsilon"],
        "pgd_steps": result["pgd_steps"],
        "images": result["images"],
        "clean_accuracy": result["clean_accuracy"],
        "fgsm_accuracy": result["fgsm_accuracy"],
        "pgd_accuracy": result["pgd_accuracy"],
        "target": threshold,
        "target_met": target_met,
    }
    summary_frame = pd.DataFrame([summary])
    per_class_frame = pd.DataFrame(result["per_class"])
    summary_frame.to_csv(output_dir / "summary.csv", index=False)
    per_class_frame.to_csv(output_dir / "per_class.csv", index=False)

    payload = {"summary": summary, "per_class": result["per_class"]}
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    title = f"{model_name} ({mode})  |  epsilon {result['epsilon']:.5f}  |  PGD steps {result['pgd_steps']}"
    plot_overall(summary_frame, output_dir / "overall.png", title)
    plot_per_class(per_class_frame, output_dir / "per_class.png", f"All CIFAR-10 classes  |  {title}")

    if save_examples_from is not None:
        model, dataset, device = save_examples_from
        save_example_grid(
            model,
            dataset,
            class_names,
            device,
            result["epsilon"],
            result["pgd_steps"],
            output_dir / "attack_examples.png",
        )
    return summary


def flag_below_threshold(rows: list[dict], threshold: float) -> list[dict]:
    """Mark every row whose FGSM or PGD accuracy is under ``threshold``."""

    flagged = []
    for row in rows:
        marked = dict(row)
        marked["threshold"] = threshold
        marked["fgsm_below_threshold"] = bool(row["fgsm_accuracy"] < threshold)
        marked["pgd_below_threshold"] = bool(row["pgd_accuracy"] < threshold)
        flagged.append(marked)
    return flagged


def evaluate_sweep(
    model: torch.nn.Module,
    loader: DataLoader,
    epsilons: list[float],
    pgd_steps: int,
    device: torch.device,
    *,
    threshold: float = ROBUST_ACCURACY_TARGET,
    class_names: tuple[str, ...] | list[str] = CIFAR10_CLASSES,
    max_batches: int | None = None,
    on_progress=None,
) -> list[dict]:
    """Clean, FGSM, and PGD accuracy at each epsilon on the same images.

    ``on_progress(done, total)`` is called after each epsilon.
    """

    if not epsilons:
        raise ValueError("Give at least one epsilon.")
    rows = []
    for index, epsilon in enumerate(sorted(epsilons)):
        result = evaluate_loader(
            model,
            loader,
            epsilon=epsilon,
            pgd_steps=pgd_steps,
            device=device,
            class_names=class_names,
            max_batches=max_batches,
            desc=f"eps {epsilon * 255:.1f}/255",
        )
        rows.append(result)
        if on_progress is not None:
            on_progress(index + 1, len(epsilons))
    return flag_below_threshold(rows, threshold)


def export_results(rows: list[dict], path: str | Path) -> Path:
    """Write evaluation rows to ``.csv`` (per-class detail dropped) or ``.json``."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".json":
        output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    elif output.suffix == ".csv":
        flat = [{key: value for key, value in row.items() if key != "per_class"} for row in rows]
        pd.DataFrame(flat).to_csv(output, index=False)
    else:
        raise ValueError("Export path must end in .csv or .json")
    return output


def format_report(summary: dict, per_class: list[dict]) -> str:
    """Plain-text table for the terminal and the training log."""

    threshold = float(summary.get("target", ROBUST_ACCURACY_TARGET))
    lines = [
        (
            f"Images {summary['images']}  |  epsilon {summary['epsilon']:.5f} "
            f"({summary['epsilon'] * 255:.2f}/255)  |  PGD steps {summary['pgd_steps']}"
        ),
        f"Clean {summary['clean_accuracy']:.2%}   FGSM {summary['fgsm_accuracy']:.2%}   "
        f"PGD {summary['pgd_accuracy']:.2%}",
    ]
    if summary["pgd_accuracy"] >= threshold:
        lines.append(f"Target met: PGD accuracy is at least {threshold:.0%}.")
    else:
        lines.append(f"Target not met: PGD accuracy is below {threshold:.0%}.")
    frame = pd.DataFrame(per_class)
    percent = frame.copy()
    for column in ("clean_accuracy", "fgsm_accuracy", "pgd_accuracy"):
        percent[column] = percent[column].map(lambda value: f"{value:.2%}")
    lines.append(percent.to_string(index=False))
    return "\n".join(lines)


def run_evaluation(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    epsilon: float | None = None,
    pgd_steps: int = 10,
    batch_size: int = 128,
    max_per_class: int | None = None,
    data_dir: str = "data",
    seed: int = 42,
    num_workers: int = 0,
    save_examples: bool = True,
    max_batches: int | None = None,
    threshold: float = ROBUST_ACCURACY_TARGET,
) -> dict:
    """Evaluate a saved model on the CIFAR-10 test set and write the report."""

    output_path = Path(output_dir)
    device = get_device()
    model, checkpoint_data, device = restore_model(checkpoint, device)
    config = checkpoint_data["config"]
    if epsilon is None:
        epsilon = float(config["epsilon"])

    print(
        f"Evaluating {config['model']} ({config['mode']}) on {device_label(device)} "
        f"with {parameter_count(model):,} parameters.",
        flush=True,
    )
    loader, full_dataset = make_test_loader(
        data_dir=data_dir,
        image_size=int(config["image_size"]),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        max_per_class=max_per_class,
        seed=seed,
    )
    result = evaluate_loader(
        model,
        loader,
        epsilon=epsilon,
        pgd_steps=pgd_steps,
        device=device,
        class_names=config.get("class_names", CIFAR10_CLASSES),
        max_batches=max_batches,
        desc="Test set",
    )
    class_names = list(config.get("class_names", CIFAR10_CLASSES))
    summary = write_evaluation_files(
        result,
        output_path,
        model_name=config["model"],
        mode=config["mode"],
        class_names=class_names,
        save_examples_from=(model, full_dataset, device) if save_examples else None,
        threshold=threshold,
    )
    print(format_report(summary, result["per_class"]), flush=True)
    print(f"Wrote {output_path}", flush=True)
    return {"summary": summary, "per_class": result["per_class"]}
