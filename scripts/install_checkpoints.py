"""Copy downloaded best.pt files into checkpoints/<run>/ so the app can attack them.

Usage:
    python scripts/install_checkpoints.py ~/Downloads/Attack2Defense_results
    python scripts/install_checkpoints.py ~/Downloads/Attack2Defense_results.zip

A run is recognised by the folder that holds its weights
(``resnet18_pgd/best.pt``) or by a file name that starts with the run name
(``resnet18_pgd_best.pt``). When a run has no config.json next to its weights,
the one from results/ is copied over. Each file is loaded once to check it is
a real training checkpoint before it is installed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attack2defense.setup_check import REPORTED_RUNS  # noqa: E402


def _show(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _run_name(weights: Path, known: tuple[str, ...]) -> str | None:
    if weights.name == "best.pt":
        return weights.parent.name
    for run in known:
        if weights.stem.startswith(run):
            return run
    return None


def _validate(weights: Path) -> str | None:
    import torch

    from attack2defense.utils import load_checkpoint

    try:
        payload = load_checkpoint(weights, torch.device("cpu"))
    except Exception as exc:  # noqa: BLE001 - any load failure means "not a checkpoint"
        return f"could not be loaded ({exc.__class__.__name__})"
    if not isinstance(payload, dict) or "backbone" not in payload or "config" not in payload:
        return "is not an Attack2Defense training checkpoint (no backbone/config)"
    return None


def _find_config(run: str, source_dir: Path, results_roots: list[Path]) -> Path | None:
    candidates = [source_dir / "config.json"]
    for root in results_roots:
        candidates += sorted(root.rglob(f"{run}/config.json"))
    return next((path for path in candidates if path.exists()), None)


def install(source: Path, destination: Path, results_roots: list[Path], force: bool) -> int:
    result_runs = {
        path.parent.name for root in results_roots if root.exists() for path in root.rglob("config.json")
    }
    known = tuple(sorted(set(REPORTED_RUNS) | result_runs, key=len, reverse=True))
    found = [path for path in sorted(source.rglob("*.pt")) if path.name != "last.pt"]
    installed = 0
    for weights in found:
        run = _run_name(weights, known)
        if run is None:
            print(f"  skip  {weights.name}: cannot tell which run it belongs to")
            continue
        target = destination / run / "best.pt"
        if target.exists() and not force:
            print(f"  skip  {run}: {_show(target)} already exists (use --force to replace)")
            continue
        problem = _validate(weights)
        if problem:
            print(f"  skip  {weights}: {problem}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(weights, target)
        config = _find_config(run, weights.parent, results_roots)
        if config is not None and not (target.parent / "config.json").exists():
            shutil.copy2(config, target.parent / "config.json")
        size = target.stat().st_size / 1024 / 1024
        print(f"  ok    {run}: {_show(target)} ({size:.1f} MB)")
        installed += 1
    return installed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="Folder or .zip that contains the downloaded best.pt files.")
    parser.add_argument("--destination", default=str(ROOT / "checkpoints"))
    parser.add_argument("--results-root", action="append", default=None, help="Where config.json files live.")
    parser.add_argument("--force", action="store_true", help="Replace weights that are already installed.")
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    if not source.exists():
        parser.error(f"{source} does not exist")
    destination = Path(args.destination)
    results_roots = [Path(path) for path in (args.results_root or [ROOT / "results"])]

    print(f"Installing checkpoints from {source}")
    if source.suffix == ".zip":
        with tempfile.TemporaryDirectory() as folder:
            with zipfile.ZipFile(source) as archive:
                archive.extractall(folder)
            installed = install(Path(folder), destination, results_roots, args.force)
    else:
        installed = install(source, destination, results_roots, args.force)

    if installed:
        print(f"\nInstalled {installed} checkpoint(s). Restart the API or wait 10 s, then refresh the app.")
    else:
        print("\nNothing installed. Expected files such as resnet18_pgd/best.pt.")
    print("Check the result with: python -m attack2defense doctor")


if __name__ == "__main__":
    main()
