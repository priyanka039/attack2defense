"""Optional extra attacks used as a benchmark, not as the main evaluation.

CW (Carlini and Wagner, L2) and AutoAttack (Croce and Hein) come from the
``torchattacks`` package. They are slower than FGSM/PGD and are not required
for the must-have empirical sweep.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from attack2defense.evaluate import restore_model
from attack2defense.utils import device_label, get_device, progress

EXTRA_ATTACKS = ("cw", "autoattack")


def _torchattacks():
    try:
        import torchattacks
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torchattacks is not installed. Run: python -m pip install torchattacks") from exc
    return torchattacks


def _make_attack(name: str, model: torch.nn.Module, epsilon: float, steps: int):
    torchattacks = _torchattacks()
    if name == "cw":
        return torchattacks.CW(model, c=1, kappa=0, steps=steps, lr=0.01)
    if name == "autoattack":
        return torchattacks.AutoAttack(
            model, norm="Linf", eps=epsilon, version="standard", n_classes=10, seed=0
        )
    raise ValueError(f"Unknown extra attack: {name}")


def evaluate_extra_attacks(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    epsilon: float,
    attacks: tuple[str, ...] | list[str] = EXTRA_ATTACKS,
    cw_steps: int = 20,
    device: torch.device | None = None,
) -> list[dict]:
    """Run each extra attack on ``loader`` and return one accuracy row per attack."""

    device = device or next(model.parameters()).device
    model.eval()
    rows = []
    for name in attacks:
        if name not in EXTRA_ATTACKS:
            raise ValueError(f"attack must be one of {EXTRA_ATTACKS}, not {name!r}")
        attack = _make_attack(name, model, epsilon, cw_steps)
        if hasattr(attack, "set_device"):
            attack.set_device(str(device))
        correct = 0
        seen = 0
        started = time.perf_counter()
        for images, labels in progress(loader, desc=name):
            images = images.to(device)
            labels = labels.to(device)
            adversarial = attack(images, labels)
            with torch.no_grad():
                correct += int(model(adversarial).argmax(dim=1).eq(labels).sum().item())
            seen += int(labels.shape[0])
        rows.append(
            {
                "attack": name,
                "norm": "L2" if name == "cw" else "Linf",
                "epsilon": None if name == "cw" else epsilon,
                "accuracy": correct / seen if seen else float("nan"),
                "images": seen,
                "seconds": time.perf_counter() - started,
            }
        )
    return rows


def run_extra_attacks(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    attacks: tuple[str, ...] | list[str] = ("cw",),
    epsilon: float | None = None,
    cw_steps: int = 20,
    batch_size: int = 32,
    max_images: int = 100,
    data_dir: str = "data",
    seed: int = 42,
    num_workers: int = 0,
) -> list[dict]:
    """Evaluate extra attacks on a class-balanced test subset and write CSV/JSON."""

    from attack2defense.data import make_test_loader

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model, checkpoint_data, device = restore_model(checkpoint, device)
    config = checkpoint_data["config"]
    if epsilon is None:
        epsilon = float(config.get("epsilon", 8 / 255))
    per_class = max(1, max_images // 10)
    print(
        f"Extra attacks on {config['model']} ({config['mode']}) / {device_label(device)}: "
        + ", ".join(attacks),
        flush=True,
    )
    loader, _ = make_test_loader(
        data_dir=data_dir,
        image_size=int(config["image_size"]),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        max_per_class=per_class,
        seed=seed,
        dataset=str(config.get("dataset", "cifar10")),
    )
    rows = evaluate_extra_attacks(
        model, loader, epsilon=epsilon, attacks=tuple(attacks), cw_steps=cw_steps, device=device
    )
    for row in rows:
        row["model"] = config["model"]
        row["mode"] = config["mode"]
    pd.DataFrame(rows).to_csv(output / "extra_attacks.csv", index=False)
    (output / "extra_attacks.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {output / 'extra_attacks.csv'}", flush=True)
    return rows
