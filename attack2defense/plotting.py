"""Save the figures the training and evaluation commands write to disk."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

CLEAN_COLOR = "#4C78A8"
FGSM_COLOR = "#F58518"
PGD_COLOR = "#E45756"
TARGET_COLOR = "#333333"


def _style() -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 11,
        }
    )


def _percent(series: pd.Series) -> pd.Series:
    return series.astype(float) * 100.0


def plot_overall(summary: pd.DataFrame, path: Path, title: str) -> None:
    """Bar chart of clean, FGSM, and PGD accuracy for one or more runs."""

    _style()
    frame = summary.copy()
    if "label" not in frame.columns:
        frame["label"] = frame.apply(
            lambda row: f"{row['model']} ({row['mode']})",
            axis=1,
        )

    index = frame["label"].tolist()
    plot_frame = pd.DataFrame(
        {
            "Clean": _percent(frame["clean_accuracy"]).tolist(),
            "FGSM": _percent(frame["fgsm_accuracy"]).tolist(),
            "PGD": _percent(frame["pgd_accuracy"]).tolist(),
        },
        index=index,
    )

    axis = plot_frame.plot(
        kind="bar",
        figsize=(max(8, 1.6 * len(index) + 4), 5),
        color=[CLEAN_COLOR, FGSM_COLOR, PGD_COLOR],
        width=0.8,
    )
    axis.axhline(80, color=TARGET_COLOR, linestyle="--", linewidth=1, label="80% target")
    axis.set_ylabel("Accuracy (%)")
    axis.set_xlabel("")
    axis.set_title(title)
    axis.set_ylim(0, 110)
    axis.set_xticks(range(len(index)))
    axis.set_xticklabels(index, rotation=0)
    axis.legend()
    for container in axis.containers:
        try:
            axis.bar_label(container, fmt="%.1f", fontsize=8, padding=2)
        except (AttributeError, ValueError):
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = axis.get_figure()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_per_class(frame: pd.DataFrame, path: Path, title: str) -> None:
    """Grouped bars for every CIFAR-10 class."""

    _style()
    plot_frame = pd.DataFrame(
        {
            "Clean": _percent(frame["clean_accuracy"]).tolist(),
            "FGSM": _percent(frame["fgsm_accuracy"]).tolist(),
            "PGD": _percent(frame["pgd_accuracy"]).tolist(),
        },
        index=frame["class"].tolist(),
    )
    axis = plot_frame.plot(
        kind="bar",
        figsize=(12, 5),
        color=[CLEAN_COLOR, FGSM_COLOR, PGD_COLOR],
    )
    axis.axhline(80, color=TARGET_COLOR, linestyle="--", linewidth=1, label="80% target")
    axis.set_ylabel("Accuracy (%)")
    axis.set_xlabel("")
    axis.set_title(title)
    axis.set_ylim(0, 110)
    axis.set_xticks(range(len(plot_frame.index)))
    axis.set_xticklabels(list(plot_frame.index), rotation=30, ha="right")
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = axis.get_figure()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_history(frame: pd.DataFrame, path: Path) -> None:
    """Training loss and validation accuracy across epochs."""

    _style()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(frame["epoch"], frame["train_loss"], color=CLEAN_COLOR, marker="o")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        frame["epoch"],
        _percent(frame["val_clean"]),
        color=CLEAN_COLOR,
        marker="o",
        label="Clean",
    )
    axes[1].plot(
        frame["epoch"],
        _percent(frame["val_fgsm"]),
        color=FGSM_COLOR,
        marker="o",
        label="FGSM",
    )
    axes[1].plot(
        frame["epoch"],
        _percent(frame["val_pgd"]),
        color=PGD_COLOR,
        marker="o",
        label="PGD",
    )
    axes[1].axhline(80, color=TARGET_COLOR, linestyle="--", linewidth=1, label="80% target")
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_attack_examples(
    original: torch.Tensor,
    fgsm: torch.Tensor,
    pgd: torch.Tensor,
    labels: list[int],
    clean_pred: list[int],
    fgsm_pred: list[int],
    pgd_pred: list[int],
    class_names: list[str],
    path: Path,
    epsilon: float,
) -> None:
    """One test image per class (columns) as clean, FGSM, PGD, and PGD noise (rows).

    Titles are the model's prediction, green when correct and red when wrong.
    The last row shows the PGD perturbation rescaled to the full colour range,
    because at 2/255 it is invisible otherwise.
    """

    _style()
    count = len(labels)
    figure, axes = plt.subplots(4, count, figsize=(1.45 * count, 7.0), squeeze=False)
    rows = (
        ("Clean", original, clean_pred),
        ("FGSM", fgsm, fgsm_pred),
        ("PGD", pgd, pgd_pred),
    )
    noise = (pgd - original).detach().cpu()
    scale = noise.abs().max().clamp_min(1e-12)

    for column, true_label in enumerate(labels):
        for row, (heading, batch, predictions) in enumerate(rows):
            axis = axes[row, column]
            axis.imshow(batch[column].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1))
            predicted = predictions[column]
            correct = predicted == true_label
            axis.set_title(class_names[predicted], color="#1b7f3a" if correct else "#b00020", fontsize=8)
            if row == 0:
                axis.set_title(
                    f"[{class_names[true_label]}]\n{class_names[predicted]}",
                    color="#1b7f3a" if correct else "#b00020",
                    fontsize=8,
                )
            if column == 0:
                axis.set_ylabel(heading, fontsize=10)
        noise_axis = axes[3, column]
        noise_axis.imshow((noise[column] / (2 * scale) + 0.5).permute(1, 2, 0).numpy().clip(0, 1))
        if column == 0:
            noise_axis.set_ylabel("PGD noise\n(amplified)", fontsize=10)

    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"Predictions before and after attack, epsilon {epsilon * 255:.0f}/255  "
        "(true class in brackets; green = correct, red = wrong)",
        fontsize=11,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_epsilon_sweep(frame: pd.DataFrame, path: Path, title: str) -> None:
    """Accuracy against a few attack budgets."""

    _style()
    figure, axis = plt.subplots(figsize=(8, 5))
    epsilons = frame["epsilon"].astype(float)
    axis.plot(epsilons, _percent(frame["clean_accuracy"]), marker="o", color=CLEAN_COLOR, label="Clean")
    axis.plot(epsilons, _percent(frame["fgsm_accuracy"]), marker="o", color=FGSM_COLOR, label="FGSM")
    axis.plot(epsilons, _percent(frame["pgd_accuracy"]), marker="o", color=PGD_COLOR, label="PGD")
    axis.axhline(80, color=TARGET_COLOR, linestyle="--", linewidth=1, label="80% target")
    axis.set_xlabel("Epsilon (pixel space, images in [0, 1])")
    axis.set_ylabel("Accuracy (%)")
    axis.set_title(title)
    axis.set_ylim(0, 105)
    axis.grid(True, alpha=0.3)
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
