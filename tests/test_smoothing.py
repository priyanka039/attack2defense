"""Randomized smoothing: noise, Clopper-Pearson bound, certified radius."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from attack2defense.models import build_pixel_model
from attack2defense.smoothing import (
    add_gaussian_noise,
    certified_radius,
    certify_loader,
    clopper_pearson_lower,
    write_certify_files,
)
from attack2defense.train import TrainConfig, _training_inputs, attack_tag


class SmoothingMathTests(unittest.TestCase):
    def test_clopper_pearson_is_between_zero_and_the_mean(self):
        self.assertEqual(clopper_pearson_lower(0, 100), 0.0)
        bound = clopper_pearson_lower(90, 100, alpha=0.001)
        self.assertGreater(bound, 0.7)
        self.assertLess(bound, 0.90)

    def test_radius_is_zero_until_p_a_exceeds_one_half(self):
        self.assertEqual(certified_radius(0.5, 0.25), 0.0)
        self.assertGreater(certified_radius(0.9, 0.25), certified_radius(0.8, 0.25))
        self.assertAlmostEqual(certified_radius(0.9, 0.5) / certified_radius(0.9, 0.25), 2.0, places=4)

    def test_noise_stays_in_unit_interval(self):
        images = torch.full((4, 3, 8, 8), 0.5)
        noisy = add_gaussian_noise(images, 0.25)
        self.assertEqual(noisy.shape, images.shape)
        self.assertGreaterEqual(float(noisy.min()), 0.0)
        self.assertLessEqual(float(noisy.max()), 1.0)
        self.assertFalse(torch.equal(noisy, images))


class GaussianTrainingTests(unittest.TestCase):
    def test_gaussian_mode_perturbs_the_batch(self):
        torch.manual_seed(0)
        model = build_pixel_model("small_cnn").eval()
        images = torch.rand(8, 3, 32, 32)
        labels = torch.arange(8) % 10
        config = TrainConfig(mode="gaussian", sigma=0.25)
        noisy = _training_inputs(model, images, labels, config)
        self.assertEqual(noisy.shape, images.shape)
        self.assertFalse(torch.equal(noisy, images))
        self.assertEqual(attack_tag("gaussian", 2 / 255, sigma=0.25), "gaussian_sigma0.25")


class CertifyTests(unittest.TestCase):
    def test_constant_model_certifies_a_positive_radius(self):
        class Constant(torch.nn.Module):
            def forward(self, pixels: torch.Tensor) -> torch.Tensor:
                logits = torch.zeros(pixels.shape[0], 10, device=pixels.device)
                logits[:, 3] = 10.0
                return logits

        images = torch.rand(6, 3, 8, 8)
        labels = torch.full((6,), 3)
        loader = DataLoader(TensorDataset(images, labels), batch_size=6)
        model = Constant().eval()
        result = certify_loader(model, loader, sigma=0.25, n0=20, n=80, batch_size=20, radii=[0.0, 0.25, 1.0])
        self.assertEqual(result["images"], 6)
        self.assertEqual(result["naive_accuracy"], 1.0)
        self.assertGreater(result["curve"][0]["certified_accuracy"], 0.5)
        self.assertGreaterEqual(
            result["curve"][0]["certified_accuracy"], result["curve"][-1]["certified_accuracy"]
        )
        with tempfile.TemporaryDirectory() as folder:
            write_certify_files(result, folder, model_name="const", mode="gaussian")
            self.assertTrue((Path(folder) / "certify.csv").exists())
            self.assertTrue((Path(folder) / "certified_accuracy.png").exists())


if __name__ == "__main__":
    unittest.main()
