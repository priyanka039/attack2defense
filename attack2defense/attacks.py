"""FGSM and PGD attacks measured in pixel space.

The model wrapper accepts images in [0, 1] and normalizes them internally.
The attack therefore edits real pixel intensities, then projects each pixel
back into an L-infinity ball of radius ``epsilon`` and into the valid range
[0, 1].

References:
- Goodfellow, Shlens, Szegedy. Explaining and Harnessing Adversarial Examples.
- Madry et al. Towards Deep Learning Models Resistant to Adversarial Attacks.
- Wong, Rice, Kolter. Fast is better than free (one-step adversarial training).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def project(images: torch.Tensor, original: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Keep ``images`` inside the L-infinity ball and inside [0, 1]."""

    lower = original - epsilon
    upper = original + epsilon
    clipped = torch.maximum(torch.minimum(images, upper), lower)
    return torch.clamp(clipped, 0.0, 1.0)


def pgd_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    random_start: bool = True,
) -> torch.Tensor:
    """Untargeted PGD. Maximizes cross-entropy of ``labels`` inside the ball.

    ``labels`` are the classes the attack tries to leave. For a trained
    CIFAR-10 model those are the true labels. For the ImageNet zero-shot
    benchmark they are the model's current predictions.
    """

    original = images.detach()
    if epsilon <= 0 or steps <= 0:
        return original

    if random_start:
        noise = torch.empty_like(original).uniform_(-epsilon, epsilon)
        adversarial = project(original + noise, original, epsilon)
    else:
        adversarial = original.clone()

    with torch.enable_grad():
        for _ in range(steps):
            adversarial = adversarial.detach().requires_grad_(True)
            # Summing keeps per-pixel gradients large enough to survive float16
            # under mixed precision. Only the sign is used, so the scale is free.
            loss = F.cross_entropy(model(adversarial), labels, reduction="sum")
            gradient = torch.autograd.grad(loss, adversarial)[0]
            adversarial = project(
                adversarial.detach() + alpha * gradient.sign().to(original.dtype),
                original,
                epsilon,
            )

    return adversarial.detach()


def fgsm_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Single signed step of size ``epsilon``, starting from the clean image."""

    return pgd_attack(
        model,
        images,
        labels,
        epsilon=epsilon,
        alpha=epsilon,
        steps=1,
        random_start=False,
    )


def fast_adversarial_example(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """One-step attack used by fast adversarial training.

    A random point inside the ball plus a step of ``1.25 * epsilon`` is the
    Wong et al. recipe. The extra step length is projected back into the ball,
    which avoids the failure mode where a plain FGSM step is too weak.
    """

    return pgd_attack(
        model,
        images,
        labels,
        epsilon=epsilon,
        alpha=1.25 * epsilon,
        steps=1,
        random_start=True,
    )


def craft_adversarial_examples(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    kind: str,
    steps: int = 10,
) -> torch.Tensor:
    """Build a batch of adversarial images.

    ``kind`` is ``fgsm`` (evaluation), ``fgsm_rs`` (fast adversarial training),
    or ``pgd`` (multi-step evaluation and PGD training).

    PGD uses the step size from Madry et al., ``2.5 * epsilon / steps``, so the
    attack can cross the whole ball whatever the step count. Ten steps give
    ``epsilon / 4``.
    """

    if kind == "fgsm":
        return fgsm_attack(model, images, labels, epsilon)
    if kind == "fgsm_rs":
        return fast_adversarial_example(model, images, labels, epsilon)
    if kind == "pgd":
        alpha = 2.5 * epsilon / max(steps, 1)
        return pgd_attack(
            model,
            images,
            labels,
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            random_start=True,
        )
    raise ValueError(f"Unknown attack kind: {kind}")
