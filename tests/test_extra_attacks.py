"""Optional torchattacks CW/AutoAttack wrappers, plus Imagenette helpers."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from attack2defense.config import DATASET_NAMES, IMAGENETTE_CLASSES
from attack2defense.data import dataset_classes, default_image_size, make_transforms

try:
    import torchattacks
except ImportError:  # pragma: no cover
    torchattacks = None


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.fc = nn.Linear(4, 10)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.relu(self.conv(pixels)).mean(dim=(2, 3)))


class DatasetHelperTests(unittest.TestCase):
    def test_imagenette_metadata(self):
        self.assertEqual(dataset_classes("imagenette"), IMAGENETTE_CLASSES)
        self.assertEqual(default_image_size("imagenette", 32), 128)
        self.assertEqual(default_image_size("cifar10", 32), 32)
        with self.assertRaises(ValueError):
            dataset_classes("not-a-dataset")
        self.assertIn("imagenette", DATASET_NAMES)

    def test_imagenette_transform_resizes(self):
        transform = make_transforms(128, train=False, dataset="imagenette")
        from PIL import Image

        image = Image.new("RGB", (160, 160), color=(20, 40, 60))
        tensor = transform(image)
        self.assertEqual(tuple(tensor.shape), (3, 128, 128))


@unittest.skipIf(torchattacks is None, "torchattacks is not installed")
class ExtraAttackTests(unittest.TestCase):
    def test_cw_runs_on_a_tiny_model(self):
        from attack2defense.extra_attacks import evaluate_extra_attacks

        torch.manual_seed(0)
        model = Tiny().eval()
        images = torch.rand(4, 3, 16, 16)
        labels = torch.arange(4)
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        rows = evaluate_extra_attacks(
            model, loader, epsilon=8 / 255, attacks=("cw",), cw_steps=3, device=torch.device("cpu")
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["attack"], "cw")
        self.assertEqual(rows[0]["images"], 4)
        self.assertGreaterEqual(rows[0]["accuracy"], 0.0)
        self.assertLessEqual(rows[0]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
