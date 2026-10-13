"""Cross-check FGSM and PGD against the torchattacks reference implementations."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from attack2defense.attacks import fgsm_attack, pgd_attack, run_attack

try:
    import torchattacks
except ImportError:  # pragma: no cover - optional dev dependency
    torchattacks = None


def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(4),
        nn.Flatten(),
        nn.Linear(8 * 16, 10),
    )
    return model.eval()


def _batch(count: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    images = torch.rand(count, 3, 32, 32, generator=generator)
    labels = torch.arange(count) % 10
    return images, labels


@unittest.skipIf(torchattacks is None, "torchattacks is not installed")
class ReferenceAttackTests(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_model()
        self.images, self.labels = _batch()
        self.epsilon = 8 / 255

    def test_fgsm_matches_torchattacks(self):
        ours = fgsm_attack(self.model, self.images, self.labels, self.epsilon)
        reference = torchattacks.FGSM(self.model, eps=self.epsilon)(self.images, self.labels)
        self.assertLess(float((ours - reference).abs().max()), 1e-6)

    def test_pgd_matches_torchattacks_without_random_start(self):
        alpha, steps = 2 / 255, 10
        ours = pgd_attack(
            self.model, self.images, self.labels, self.epsilon, alpha, steps, random_start=False
        )
        reference = torchattacks.PGD(
            self.model, eps=self.epsilon, alpha=alpha, steps=steps, random_start=False
        )(self.images, self.labels)
        # Projection order differs by a rounding step, so allow float noise.
        self.assertLess(float((ours - reference).abs().mean()), 1e-4)
        with torch.no_grad():
            ours_pred = self.model(ours).argmax(dim=1)
            reference_pred = self.model(reference).argmax(dim=1)
        self.assertGreaterEqual(float(ours_pred.eq(reference_pred).float().mean()), 0.9)


class AttackTraceTests(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_model()
        self.images, self.labels = _batch()

    def test_pgd_trace_has_one_entry_per_step(self):
        result = run_attack(self.model, self.images, self.labels, kind="pgd", epsilon=8 / 255, steps=5)
        self.assertEqual(result.steps, 5)
        self.assertEqual(len(result.success_per_step), 5)
        self.assertTrue(all(0.0 <= rate <= 1.0 for rate in result.success_per_step))
        self.assertAlmostEqual(result.alpha, 2.5 * (8 / 255) / 5)
        self.assertGreaterEqual(result.seconds, 0.0)
        perturbation = (result.adversarial - self.images).abs().max()
        self.assertLessEqual(float(perturbation), 8 / 255 + 1e-6)

    def test_fgsm_is_one_step_without_random_start(self):
        result = run_attack(self.model, self.images, self.labels, kind="fgsm", epsilon=4 / 255, steps=7)
        self.assertEqual((result.steps, result.random_start, result.alpha), (1, False, 4 / 255))
        self.assertEqual(len(result.success_per_step), 1)

    def test_invalid_attack_settings_raise(self):
        with self.assertRaises(ValueError):
            run_attack(self.model, self.images, self.labels, kind="cw", epsilon=0.01)
        with self.assertRaises(ValueError):
            run_attack(self.model, self.images, self.labels, kind="pgd", epsilon=2.0)
        with self.assertRaises(ValueError):
            run_attack(self.model, self.images, self.labels, kind="pgd", epsilon=0.01, steps=0)


if __name__ == "__main__":
    unittest.main()
