"""Checks that attacks stay inside the pixel ball and that names match."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from attack2defense.attacks import craft_adversarial_examples, fgsm_attack, pgd_attack
from attack2defense.cli import parse_epsilon
from attack2defense.data import stratified_indices
from attack2defense.models import PixelModel
from attack2defense.zeroshot import is_cat_category


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8, 10)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.conv(pixels)).mean(dim=(2, 3))
        return self.fc(hidden)


class AttackTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = PixelModel(Tiny(), mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
        self.model.eval()
        self.images = torch.rand(4, 3, 16, 16) * 0.5 + 0.25
        self.labels = torch.randint(0, 10, (4,))
        self.epsilon = 0.03

    def _assert_inside_ball(self, adversarial: torch.Tensor) -> None:
        delta = (adversarial - self.images).abs().max().item()
        self.assertLessEqual(delta, self.epsilon + 1e-5)
        self.assertGreaterEqual(adversarial.min().item(), 0.0)
        self.assertLessEqual(adversarial.max().item(), 1.0)

    def test_pgd_respects_the_pixel_ball(self):
        adversarial = pgd_attack(
            self.model,
            self.images,
            self.labels,
            epsilon=self.epsilon,
            alpha=self.epsilon / 4,
            steps=5,
            random_start=True,
        )
        self._assert_inside_ball(adversarial)
        self.assertFalse(torch.allclose(adversarial, self.images))

    def test_fgsm_respects_the_pixel_ball(self):
        adversarial = fgsm_attack(self.model, self.images, self.labels, self.epsilon)
        self._assert_inside_ball(adversarial)

    def test_fast_training_attack_respects_the_pixel_ball(self):
        adversarial = craft_adversarial_examples(
            self.model,
            self.images,
            self.labels,
            self.epsilon,
            kind="fgsm_rs",
        )
        self._assert_inside_ball(adversarial)

    def test_zero_epsilon_returns_the_original(self):
        adversarial = pgd_attack(
            self.model,
            self.images,
            self.labels,
            epsilon=0.0,
            alpha=0.0,
            steps=3,
        )
        self.assertTrue(torch.equal(adversarial, self.images))

    def test_small_fgsm_step_does_not_decrease_loss(self):
        epsilon = 1e-3
        images = torch.full((2, 3, 8, 8), 0.5)
        labels = torch.tensor([1, 3])
        before = torch.nn.functional.cross_entropy(self.model(images), labels)
        adversarial = fgsm_attack(self.model, images, labels, epsilon)
        after = torch.nn.functional.cross_entropy(self.model(adversarial), labels)
        self.assertGreaterEqual(after.item() + 1e-6, before.item())

    def test_pixel_model_normalizes(self):
        class Identity(nn.Module):
            def forward(self, pixels: torch.Tensor) -> torch.Tensor:
                return pixels

        model = PixelModel(Identity(), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        output = model(torch.full((1, 3, 2, 2), 0.5))
        self.assertTrue(torch.allclose(output, torch.zeros_like(output)))

    def test_validation_split_covers_every_class(self):
        targets = [index % 10 for index in range(100)]
        train_indices, val_indices = stratified_indices(targets, val_per_class=2, seed=1)
        self.assertEqual(len(val_indices), 20)
        self.assertEqual(len(train_indices), 80)
        self.assertEqual(set(train_indices) & set(val_indices), set())
        self.assertEqual(set(train_indices) | set(val_indices), set(range(100)))

    def test_epsilon_fraction_and_cat_names(self):
        self.assertAlmostEqual(parse_epsilon("2/255"), 2 / 255)
        self.assertAlmostEqual(parse_epsilon("0.01"), 0.01)
        self.assertTrue(is_cat_category("tiger cat"))
        self.assertTrue(is_cat_category("Persian cat"))
        self.assertFalse(is_cat_category("catamaran"))
        self.assertFalse(is_cat_category("caterpillar"))


if __name__ == "__main__":
    unittest.main()
