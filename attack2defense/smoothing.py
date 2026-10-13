"""Randomized smoothing (Cohen, Rosenfeld, Kolter, ICML 2019).

A smoothed classifier predicts the majority class of ``f(x + N(0, sigma^2 I))``.
If that class wins with probability at least ``p_A``, every perturbation of L2
norm less than ``sigma * Phi^{-1}(p_A)`` is certified not to change the
prediction. That bound is independent of the attack used at test time.

This module is an extra evaluation track. Empirical FGSM/PGD numbers stay in
``evaluate.py`` and must not be mixed with certified accuracy.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from attack2defense.evaluate import restore_model
from attack2defense.plotting import plot_certified_accuracy
from attack2defense.utils import device_label, get_device, progress

DEFAULT_SIGMA = 0.25
DEFAULT_ALPHA = 0.001
DEFAULT_N0 = 100
DEFAULT_N = 1000


def add_gaussian_noise(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """Isotropic Gaussian noise, clipped back to a valid image."""

    if sigma <= 0:
        return images
    return torch.clamp(images + torch.randn_like(images) * sigma, 0.0, 1.0)


def _normal_ppf(probability: float) -> float:
    """Inverse CDF of the standard normal. ``probability`` must be in (0, 1)."""

    if probability <= 0.0:
        return float("-inf")
    if probability >= 1.0:
        return float("inf")
    return float(math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * probability - 1.0)))


def _log_binom(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), in log space so n ~ 10,000 is safe."""

    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0 if k < n else 1.0
    log_p = math.log(p)
    log_q = math.log(1.0 - p)
    total = 0.0
    for i in range(k + 1):
        total += math.exp(_log_binom(n, i) + i * log_p + (n - i) * log_q)
    return min(total, 1.0)


def clopper_pearson_lower(successes: int, trials: int, alpha: float = DEFAULT_ALPHA) -> float:
    """One-sided Clopper-Pearson lower bound on a binomial proportion.

    Finds the largest ``p`` such that P(X >= successes | Binomial(trials, p))
    is at least ``alpha``. Returns 0 when there are no successes.
    """

    if successes <= 0 or trials <= 0:
        return 0.0
    if successes >= trials:
        return alpha ** (1.0 / trials)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        # P(X >= k) = 1 - F(k - 1). We want this equal to alpha.
        tail = 1.0 - _binom_cdf(successes - 1, trials, mid)
        if tail >= alpha:
            hi = mid
        else:
            lo = mid
    return lo


def certified_radius(p_a: float, sigma: float) -> float:
    """L2 radius ``sigma * Phi^{-1}(p_A)``. Zero when p_A is not above 1/2."""

    if p_a <= 0.5 or sigma <= 0:
        return 0.0
    return float(sigma * _normal_ppf(p_a))


@torch.no_grad()
def _count_class(
    model: torch.nn.Module,
    image: torch.Tensor,
    sigma: float,
    samples: int,
    batch_size: int,
    class_index: int | None,
    num_classes: int,
) -> tuple[int, int]:
    """Sample the noisy classifier. Returns (winner class, votes for that class).

    When ``class_index`` is set, count only that class (the confirmation step).
    """

    votes = torch.zeros(num_classes, dtype=torch.long)
    remaining = samples
    image = image.unsqueeze(0)
    while remaining > 0:
        take = min(batch_size, remaining)
        noisy = add_gaussian_noise(image.expand(take, *image.shape[1:]), sigma)
        predicted = model(noisy).argmax(dim=1).cpu()
        votes += torch.bincount(predicted, minlength=num_classes)[:num_classes]
        remaining -= take
    if class_index is None:
        winner = int(votes.argmax())
        return winner, int(votes[winner])
    return class_index, int(votes[class_index])


@torch.no_grad()
def certify_image(
    model: torch.nn.Module,
    image: torch.Tensor,
    label: int,
    *,
    sigma: float,
    n0: int,
    n: int,
    alpha: float,
    batch_size: int,
) -> dict:
    """Certify one image. The prediction uses ``n0`` samples; the radius uses ``n``."""

    num_classes = int(model(image.unsqueeze(0)).shape[1])
    guess, _ = _count_class(model, image, sigma, n0, batch_size, None, num_classes)
    _, votes = _count_class(model, image, sigma, n, batch_size, guess, num_classes)
    p_a = clopper_pearson_lower(votes, n, alpha)
    radius = certified_radius(p_a, sigma)
    correct = guess == int(label)
    return {
        "true_label": int(label),
        "predicted": guess,
        "correct": bool(correct),
        "votes": votes,
        "samples": n,
        "p_a_lower": p_a,
        "radius": radius if correct else 0.0,
        "abstained": p_a <= 0.5,
    }


@torch.no_grad()
def certify_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    sigma: float = DEFAULT_SIGMA,
    n0: int = DEFAULT_N0,
    n: int = DEFAULT_N,
    alpha: float = DEFAULT_ALPHA,
    batch_size: int = 64,
    radii: list[float] | None = None,
    on_progress=None,
) -> dict:
    """Certify every image in ``loader``. Returns per-image rows and a radius curve.

    Certified accuracy at radius ``r`` is the share of images whose prediction
    matches the true label *and* whose certified radius is at least ``r``.
    That curve is separate from empirical robust accuracy.
    """

    if radii is None:
        radii = [round(value * 0.05, 2) for value in range(0, 21)]
    rows = []
    seen = 0
    total = len(loader.dataset) if hasattr(loader, "dataset") else None
    for images, labels in progress(loader, desc="Certify"):
        for image, label in zip(images, labels, strict=True):
            rows.append(
                certify_image(
                    model,
                    image.to(images.device),
                    int(label),
                    sigma=sigma,
                    n0=n0,
                    n=n,
                    alpha=alpha,
                    batch_size=batch_size,
                )
            )
            seen += 1
            if on_progress is not None:
                on_progress(seen, total or seen)
    curve = []
    for radius in radii:
        certified = sum(1 for row in rows if row["correct"] and row["radius"] >= radius)
        curve.append(
            {
                "radius": radius,
                "certified_accuracy": certified / len(rows) if rows else float("nan"),
                "images": len(rows),
            }
        )
    return {
        "sigma": sigma,
        "n0": n0,
        "n": n,
        "alpha": alpha,
        "images": len(rows),
        "naive_accuracy": sum(row["correct"] for row in rows) / len(rows) if rows else float("nan"),
        "rows": rows,
        "curve": curve,
    }


def write_certify_files(result: dict, output_dir: str | Path, *, model_name: str, mode: str) -> Path:
    """Write the certified-accuracy curve as CSV, JSON, and a plot."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["curve"]).to_csv(output / "certify.csv", index=False)
    payload = {key: value for key, value in result.items() if key != "rows"}
    payload["model"] = model_name
    payload["mode"] = mode
    (output / "certify.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_certified_accuracy(
        pd.DataFrame(result["curve"]),
        output / "certified_accuracy.png",
        f"{model_name} ({mode}), sigma={result['sigma']}",
    )
    return output / "certify.csv"


def run_certify(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    sigma: float = DEFAULT_SIGMA,
    n0: int = DEFAULT_N0,
    n: int = DEFAULT_N,
    alpha: float = DEFAULT_ALPHA,
    batch_size: int = 64,
    max_per_class: int | None = 20,
    data_dir: str = "data",
    seed: int = 42,
    num_workers: int = 0,
    on_progress=None,
) -> dict:
    """Load a checkpoint, certify a class-balanced test subset, and write files."""

    from attack2defense.data import make_test_loader

    device = get_device()
    model, checkpoint_data, device = restore_model(checkpoint, device)
    config = checkpoint_data["config"]
    print(f"Certifying {config['model']} ({config['mode']}) on {device_label(device)}.", flush=True)
    loader, _ = make_test_loader(
        data_dir=data_dir,
        image_size=int(config["image_size"]),
        batch_size=1,
        num_workers=num_workers,
        device=device,
        max_per_class=max_per_class,
        seed=seed,
        dataset=str(config.get("dataset", "cifar10")),
    )
    result = certify_loader(
        model,
        loader,
        sigma=sigma,
        n0=n0,
        n=n,
        alpha=alpha,
        batch_size=batch_size,
        on_progress=on_progress,
    )
    write_certify_files(result, output_dir, model_name=config["model"], mode=config["mode"])
    top = result["curve"][0]["certified_accuracy"] if result["curve"] else float("nan")
    print(f"Certified accuracy at radius 0: {top:.2%}. Wrote {output_dir}", flush=True)
    return result
