"""Bundle the source code into a zip for upload to Google Colab.

Usage:
    python scripts/package_for_colab.py

Datasets, outputs, checkpoints, and caches are left out, so the archive
stays small.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = (
    "attack2defense",
    "tests",
    "scripts",
    "notebooks",
    "comparing_models.py",
    "requirements.txt",
    "pyproject.toml",
    "README.md",
)
EXCLUDED_PARTS = {"__pycache__", ".ipynb_checkpoints", ".ruff_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pt", ".zip"}


def _files() -> list[Path]:
    found: list[Path] = []
    for entry in INCLUDE:
        path = ROOT / entry
        candidates = [path] if path.is_file() else sorted(path.rglob("*"))
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if EXCLUDED_PARTS.intersection(candidate.parts) or candidate.suffix in EXCLUDED_SUFFIXES:
                continue
            found.append(candidate)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=str(ROOT / "attack2defense_project.zip"))
    args = parser.parse_args()

    output = Path(args.output)
    files = _files()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f"Wrote {output} ({len(files)} files, {output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
