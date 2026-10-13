"""Small helpers shared by training, evaluation, and the CLI."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

import torch
from tqdm import tqdm

T = TypeVar("T")

_PROGRESS_ENABLED = True


def set_progress_enabled(enabled: bool) -> None:
    """Turn the per-batch progress bars on or off for the whole process."""

    global _PROGRESS_ENABLED
    _PROGRESS_ENABLED = enabled


def progress(iterable: Iterable[T], desc: str) -> Iterator[T]:
    """Wrap an iterable in a progress bar unless bars are disabled."""

    return tqdm(iterable, desc=desc, leave=False, mininterval=2, disable=not _PROGRESS_ENABLED)


def get_device() -> torch.device:
    """Use a CUDA GPU when one is visible, otherwise the CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_label(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    """Load a training checkpoint, including its optimizer state."""

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def make_logger(path: Path):
    """Print a line and append it to a text log."""

    path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        print(message, flush=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    return log
