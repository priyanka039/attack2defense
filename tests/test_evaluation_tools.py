"""Batch ratio, attack tags, threshold flags, export, registry, and training hooks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from attack2defense.evaluate import evaluate_sweep, export_results, flag_below_threshold
from attack2defense.models import build_pixel_model
from attack2defense.registry import discover_models
from attack2defense.train import (
    TrainConfig,
    TrainingCancelled,
    TrainingHooks,
    _make_optimizer,
    _make_scheduler,
    _training_inputs,
    attack_tag,
    build_run_config,
    train_one_epoch,
)


def _loader(count: int = 20, batch_size: int = 10) -> DataLoader:
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(count, 3, 32, 32, generator=generator)
    labels = torch.arange(count) % 10
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


class BatchRatioTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = build_pixel_model("small_cnn").eval()
        self.images = torch.rand(10, 3, 32, 32)
        self.labels = torch.arange(10)

    def test_ratio_keeps_the_clean_share_untouched(self):
        config = TrainConfig(mode="pgd", pgd_steps=2, adv_ratio=0.4, epsilon=8 / 255)
        inputs = _training_inputs(self.model, self.images, self.labels, config)
        self.assertEqual(inputs.shape, self.images.shape)
        self.assertTrue(torch.equal(inputs[4:], self.images[4:]))
        self.assertFalse(torch.equal(inputs[:4], self.images[:4]))

    def test_zero_ratio_is_clean_training(self):
        config = TrainConfig(mode="fgsm", adv_ratio=0.0)
        inputs = _training_inputs(self.model, self.images, self.labels, config)
        self.assertTrue(torch.equal(inputs, self.images))

    def test_tag_records_the_attack_config(self):
        self.assertEqual(attack_tag("clean", 2 / 255), "clean")
        self.assertEqual(attack_tag("pgd", 2 / 255, 5, 1.0), "pgd5_eps2_ratio1")
        self.assertEqual(attack_tag("fgsm", 8 / 255, 5, 0.5), "fgsm_rs_eps8_ratio0.5")
        self.assertEqual(build_run_config(TrainConfig(mode="pgd"))["tag"], "pgd5_eps2_ratio1")


class TrainingHookTests(unittest.TestCase):
    def _epoch(self, hooks: TrainingHooks) -> None:
        config = TrainConfig(mode="clean", epochs=1, optimizer_name="sgd", lr=0.01, amp=False)
        model = build_pixel_model("small_cnn")
        optimizer = _make_optimizer(model, config)
        loader = _loader()
        scheduler, per_batch = _make_scheduler(optimizer, config, len(loader))
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        train_one_epoch(
            model, loader, optimizer, scaler, scheduler, per_batch, torch.device("cpu"), config, False, hooks
        )

    def test_batch_hook_reports_progress(self):
        seen: list[tuple[int, int, int]] = []
        self._epoch(TrainingHooks(on_batch=lambda *args: seen.append(args)))
        self.assertEqual(seen, [(1, 1, 2), (1, 2, 2)])

    def test_should_stop_cancels(self):
        with self.assertRaises(TrainingCancelled):
            self._epoch(TrainingHooks(should_stop=lambda: True))


class ThresholdAndExportTests(unittest.TestCase):
    def test_sweep_flags_rows_below_threshold(self):
        model = build_pixel_model("small_cnn").eval()
        rows = evaluate_sweep(
            model,
            _loader(),
            [8 / 255, 2 / 255, 4 / 255],
            pgd_steps=2,
            device=torch.device("cpu"),
            threshold=0.9,
        )
        self.assertEqual([row["epsilon"] for row in rows], sorted([2 / 255, 4 / 255, 8 / 255]))
        self.assertTrue(all(row["pgd_below_threshold"] for row in rows))
        self.assertEqual(len(rows[0]["per_class"]), 10)

    def test_flags_and_export(self):
        rows = flag_below_threshold(
            [
                {
                    "epsilon": 0.01,
                    "fgsm_accuracy": 0.9,
                    "pgd_accuracy": 0.85,
                    "per_class": [{"class": "cat"}],
                },
                {"epsilon": 0.03, "fgsm_accuracy": 0.7, "pgd_accuracy": 0.6, "per_class": []},
            ],
            threshold=0.8,
        )
        self.assertEqual([row["pgd_below_threshold"] for row in rows], [False, True])
        self.assertEqual([row["fgsm_below_threshold"] for row in rows], [False, True])
        with tempfile.TemporaryDirectory() as folder:
            csv_path = export_results(rows, Path(folder) / "out.csv")
            json_path = export_results(rows, Path(folder) / "out.json")
            frame = pd.read_csv(csv_path)
            self.assertNotIn("per_class", frame.columns)
            self.assertEqual(len(frame), 2)
            self.assertEqual(json.loads(json_path.read_text())[0]["per_class"], [{"class": "cat"}])
            with self.assertRaises(ValueError):
                export_results(rows, Path(folder) / "out.txt")


class RegistryTests(unittest.TestCase):
    def test_merges_checkpoints_with_results_by_run_name(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "results" / "small_cnn_pgd"
            (run / "test_eval").mkdir(parents=True)
            config = build_run_config(TrainConfig(mode="pgd"))
            (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
            pd.DataFrame([{"epoch": 1, "val_pgd": 0.5}]).to_csv(run / "history.csv", index=False)
            pd.DataFrame([{"model": "small_cnn", "pgd_accuracy": 0.5}]).to_csv(
                run / "test_eval" / "summary.csv", index=False
            )

            weights = root / "checkpoints" / "small_cnn_pgd" / "best.pt"
            weights.parent.mkdir(parents=True)
            torch.save({"config": config}, weights)
            orphan = root / "checkpoints" / "small_cnn_clean" / "best.pt"
            orphan.parent.mkdir(parents=True)
            torch.save({"config": build_run_config(TrainConfig(mode="clean"))}, orphan)

            entries = discover_models([root / "checkpoints"], [root / "results"])
            self.assertEqual(sorted(entries), ["small_cnn_clean", "small_cnn_pgd"])
            pgd = entries["small_cnn_pgd"]
            self.assertEqual(pgd.checkpoint, weights)
            self.assertEqual(pgd.tag, "pgd5_eps2_ratio1")
            self.assertEqual(pgd.summary["pgd_accuracy"], 0.5)
            self.assertEqual(pgd.history(), [{"epoch": 1, "val_pgd": 0.5}])
            self.assertEqual(entries["small_cnn_clean"].to_dict()["label"], "No defense")


if __name__ == "__main__":
    unittest.main()
