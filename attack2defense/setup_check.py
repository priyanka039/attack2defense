"""Report what is on disk: the dataset, trained weights, and published results.

Used by ``python -m attack2defense doctor`` and by ``GET /api/setup`` so that a
missing ``best.pt`` is explained in one place instead of surfacing as an empty
screen.
"""

from __future__ import annotations

from pathlib import Path

from attack2defense.registry import discover_models

# The three ResNet-18 runs trained on Colab and reported in REPORT.md.
REPORTED_RUNS: tuple[str, ...] = ("resnet18_clean", "resnet18_fgsm", "resnet18_pgd")


def cifar10_present(data_dir: str | Path) -> bool:
    folder = Path(data_dir) / "cifar-10-batches-py"
    return (folder / "test_batch").exists() and (folder / "data_batch_1").exists()


def _display(path: Path, base: Path | None) -> str:
    if base is not None:
        try:
            return Path(path).resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            pass
    return Path(path).as_posix()


def inspect_setup(
    data_dir: str | Path,
    checkpoint_roots: list[str | Path] | tuple,
    results_roots: list[str | Path] | tuple,
    base: Path | None = None,
) -> dict:
    """Return a JSON-friendly description of what can and cannot run.

    Paths are shown relative to ``base`` when they live under it.
    """

    entries = discover_models(checkpoint_roots, results_roots)
    primary_root = Path(checkpoint_roots[0]) if checkpoint_roots else Path("checkpoints")

    runs = [
        {
            "id": entry.id,
            "architecture": entry.config.get("model"),
            "mode": entry.mode,
            "has_weights": entry.checkpoint is not None,
            "has_results": entry.summary is not None,
            "checkpoint": _display(entry.checkpoint, base) if entry.checkpoint else None,
        }
        for entry in entries.values()
    ]

    missing = [
        {"id": run_id, "expected_path": _display(primary_root / run_id / "best.pt", base)}
        for run_id in REPORTED_RUNS
        if entries.get(run_id) is None or entries[run_id].checkpoint is None
    ]

    attackable = sum(run["has_weights"] for run in runs)
    with_results = sum(run["has_results"] for run in runs)
    data_ok = cifar10_present(data_dir)

    checks = [
        {
            "id": "dataset",
            "label": "CIFAR-10 test images",
            "ok": data_ok,
            "hint": None if data_ok else "Downloaded automatically on first use (about 170 MB).",
        },
        {
            "id": "results",
            "label": "Published results",
            "ok": with_results > 0,
            "hint": None
            if with_results
            else "Unzip results.zip from the Colab run into results/ to show the reported numbers.",
        },
        {
            "id": "weights",
            "label": "Weights for live attacks",
            "ok": attackable > 0,
            "hint": None
            if attackable
            else "Train a quick model on the Training page, or install checkpoints.",
        },
        {
            "id": "reported_weights",
            "label": "Reported ResNet-18 weights",
            "ok": not missing,
            "hint": None
            if not missing
            else "Download best.pt for each run from Google Drive, then run "
            "python scripts/install_checkpoints.py <download folder or zip>.",
        },
    ]

    return {
        "ready": data_ok and attackable > 0,
        "checks": checks,
        "runs": runs,
        "missing_reported_weights": missing,
    }
