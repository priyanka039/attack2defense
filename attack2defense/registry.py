"""Find trained models and their results on disk.

A run is a folder named after the model, for example ``resnet18_pgd``. Its
weights (``best.pt``) and its results (``config.json``, ``history.csv``,
``test_eval/``) may live in the same folder or in two different roots, such
as ``checkpoints/resnet18_pgd/best.pt`` and ``results/resnet18_pgd/``. The
registry merges both by folder name.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch

from attack2defense.train import attack_tag
from attack2defense.utils import load_checkpoint

MODE_LABELS = {
    "clean": "No defense",
    "fgsm": "FGSM adversarial training",
    "pgd": "PGD adversarial training",
    "gaussian": "Gaussian noise training",
}


@dataclass
class ModelEntry:
    id: str
    config: dict
    checkpoint: Path | None = None
    run_dir: Path | None = None
    summary: dict | None = None
    sweep: list[dict] = field(default_factory=list)

    @property
    def mode(self) -> str:
        return str(self.config.get("mode", "unknown"))

    @property
    def tag(self) -> str:
        if "tag" in self.config:
            return str(self.config["tag"])
        return attack_tag(
            self.mode,
            float(self.config.get("epsilon", 0.0)),
            int(self.config.get("pgd_steps", 5)),
            float(self.config.get("adv_ratio", 1.0)),
            float(self.config.get("sigma", 0.25)),
        )

    @property
    def history_path(self) -> Path | None:
        if self.run_dir is None:
            return None
        path = self.run_dir / "history.csv"
        return path if path.exists() else None

    def history(self) -> list[dict]:
        path = self.history_path
        return _clean_records(pd.read_csv(path)) if path is not None else []

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "architecture": self.config.get("model"),
            "dataset": self.config.get("dataset", "cifar10"),
            "mode": self.mode,
            "label": MODE_LABELS.get(self.mode, self.mode),
            "tag": self.tag,
            "epsilon": self.config.get("epsilon"),
            "epochs": self.config.get("epochs"),
            "has_weights": self.checkpoint is not None,
            "summary": self.summary,
            "sweep": self.sweep,
        }


def _clean_records(frame: pd.DataFrame) -> list[dict]:
    records = frame.to_dict(orient="records")
    return [
        {
            key: (None if isinstance(value, float) and math.isnan(value) else value)
            for key, value in record.items()
        }
        for record in records
    ]


def _read_results(run_dir: Path) -> tuple[dict | None, list[dict]]:
    summary_path = run_dir / "test_eval" / "summary.csv"
    sweep_path = run_dir / "test_eval" / "sweep.csv"
    summary = _clean_records(pd.read_csv(summary_path))[0] if summary_path.exists() else None
    sweep = _clean_records(pd.read_csv(sweep_path)) if sweep_path.exists() else []
    return summary, sweep


def discover_models(
    checkpoint_roots: list[str | Path] | tuple = (),
    results_roots: list[str | Path] | tuple = (),
) -> dict[str, ModelEntry]:
    """Return every run found under the given roots, keyed by folder name."""

    entries: dict[str, ModelEntry] = {}

    for root in map(Path, results_roots):
        if not root.exists():
            continue
        for config_path in sorted(root.rglob("config.json")):
            run_dir = config_path.parent
            config = json.loads(config_path.read_text(encoding="utf-8"))
            summary, sweep = _read_results(run_dir)
            entries[run_dir.name] = ModelEntry(
                id=run_dir.name, config=config, run_dir=run_dir, summary=summary, sweep=sweep
            )

    for root in map(Path, checkpoint_roots):
        if not root.exists():
            continue
        for weights in sorted(root.rglob("best.pt")):
            run_id = weights.parent.name
            entry = entries.get(run_id)
            if entry is None:
                config_path = weights.parent / "config.json"
                if config_path.exists():
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                else:
                    config = load_checkpoint(weights, torch.device("cpu"))["config"]
                summary, sweep = _read_results(weights.parent)
                entry = ModelEntry(
                    id=run_id,
                    config=config,
                    run_dir=weights.parent,
                    summary=summary,
                    sweep=sweep,
                )
                entries[run_id] = entry
            entry.checkpoint = weights

    return dict(sorted(entries.items()))
