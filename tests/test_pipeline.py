"""End-to-end checks of training, checkpointing, and evaluation on random data."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from attack2defense.config import CIFAR10_CLASSES
from attack2defense.evaluate import evaluate_loader, restore_model, write_evaluation_files
from attack2defense.models import build_pixel_model
from attack2defense.train import (
    TrainConfig,
    _make_optimizer,
    _make_scheduler,
    build_run_config,
    train_one_epoch,
)


class LabelledTensors(TensorDataset):
    """TensorDataset with the ``targets`` attribute the example grid expects."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        super().__init__(images, labels)
        self.targets = labels.tolist()


def _random_cifar(count: int) -> LabelledTensors:
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(count, 3, 32, 32, generator=generator)
    labels = torch.arange(count) % len(CIFAR10_CLASSES)
    return LabelledTensors(images, labels)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cpu")
        self.dataset = _random_cifar(40)
        self.loader = DataLoader(self.dataset, batch_size=20)

    def _train(self, mode: str, scheduler_name: str) -> tuple[torch.nn.Module, TrainConfig]:
        config = TrainConfig(
            model="small_cnn",
            mode=mode,
            epochs=1,
            optimizer_name="sgd",
            scheduler_name=scheduler_name,
            lr=0.01,
            amp=False,
        )
        model = build_pixel_model(config.model)
        optimizer = _make_optimizer(model, config)
        scheduler, per_batch = _make_scheduler(optimizer, config, len(self.loader))
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        loss, accuracy = train_one_epoch(
            model, self.loader, optimizer, scaler, scheduler, per_batch, self.device, config, False
        )
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        return model, config

    def test_every_training_mode_runs(self):
        for mode in ("clean", "fgsm", "pgd"):
            for scheduler_name in ("cosine", "onecycle"):
                with self.subTest(mode=mode, scheduler=scheduler_name):
                    self._train(mode, scheduler_name)

    def test_checkpoint_round_trip_and_report_files(self):
        model, config = self._train("fgsm", "onecycle")
        model.eval()

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            checkpoint = output / "best.pt"
            torch.save(
                {"backbone": model.backbone.state_dict(), "config": build_run_config(config)},
                checkpoint,
            )
            restored, _, _ = restore_model(checkpoint, self.device)
            images = self.dataset.tensors[0][:8]
            with torch.no_grad():
                self.assertTrue(torch.allclose(model(images), restored(images), atol=1e-5))

            result = evaluate_loader(restored, self.loader, epsilon=2 / 255, pgd_steps=2, device=self.device)
            self.assertEqual(result["images"], 40)
            self.assertEqual(len(result["per_class"]), len(CIFAR10_CLASSES))
            for key in ("clean_accuracy", "fgsm_accuracy", "pgd_accuracy"):
                self.assertGreaterEqual(result[key], 0.0)
                self.assertLessEqual(result[key], 1.0)

            write_evaluation_files(
                result,
                output / "eval",
                model_name="small_cnn",
                mode="fgsm",
                class_names=list(CIFAR10_CLASSES),
                save_examples_from=(restored, self.dataset, self.device),
            )
            for name in (
                "summary.csv",
                "per_class.csv",
                "metrics.json",
                "overall.png",
                "per_class.png",
                "attack_examples.png",
                "attack_examples.csv",
            ):
                self.assertTrue((output / "eval" / name).exists(), name)
            examples = pd.read_csv(output / "eval" / "attack_examples.csv")
            self.assertEqual(list(examples["true_class"]), list(CIFAR10_CLASSES))


if __name__ == "__main__":
    unittest.main()
