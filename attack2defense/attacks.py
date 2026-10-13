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

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

ATTACK_KINDS = ("fgsm", "pgd")


@dataclass
class AttackResult:
    """Adversarial images plus the cost and progress of the attack that made them."""

    adversarial: torch.Tensor
    kind: str
    epsilon: float
    alpha: float
    steps: int
    random_start: bool
    seconds: float
    success_per_step: list[float] = field(default_factory=list)


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
    trace: list[float] | None = None,
) -> torch.Tensor:
    """Untargeted PGD. Maximizes cross-entropy of ``labels`` inside the ball.

    ``labels`` are the classes the attack tries to leave. For a trained
    CIFAR-10 model those are the true labels. For the ImageNet zero-shot
    benchmark they are the model's current predictions.

    When ``trace`` is a list, the share of images the model gets wrong after
    each step is appended to it.
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
        for step in range(steps):
            adversarial = adversarial.detach().requires_grad_(True)
            logits = model(adversarial)
            if trace is not None and step > 0:
                trace.append(_miss_rate(logits, labels))
            # Summing keeps per-pixel gradients large enough to survive float16
            # under mixed precision. Only the sign is used, so the scale is free.
            loss = F.cross_entropy(logits, labels, reduction="sum")
            gradient = torch.autograd.grad(loss, adversarial)[0]
            adversarial = project(
                adversarial.detach() + alpha * gradient.sign().to(original.dtype),
                original,
                epsilon,
            )

    if trace is not None:
        with torch.no_grad():
            trace.append(_miss_rate(model(adversarial), labels))
    return adversarial.detach()


def _miss_rate(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(logits.detach().argmax(dim=1).ne(labels).float().mean().item())


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


def run_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    kind: str,
    epsilon: float,
    alpha: float | None = None,
    steps: int = 10,
    random_start: bool = True,
) -> AttackResult:
    """Run FGSM or PGD and record the per-step success rate and the run time.

    FGSM is fixed to one step of size ``epsilon`` from the clean image. For
    PGD, ``alpha`` defaults to ``2.5 * epsilon / steps``.
    """

    if kind not in ATTACK_KINDS:
        raise ValueError(f"kind must be one of {ATTACK_KINDS}, not {kind!r}")
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be between 0 and 1")
    if kind == "fgsm":
        alpha, steps, random_start = epsilon, 1, False
    else:
        if steps < 1:
            raise ValueError("PGD needs at least one step")
        if alpha is None:
            alpha = 2.5 * epsilon / steps
        if alpha <= 0 and epsilon > 0:
            raise ValueError("alpha must be positive")

    trace: list[float] = []
    _synchronize(images.device)
    started = time.perf_counter()
    adversarial = pgd_attack(model, images, labels, epsilon, alpha, steps, random_start, trace=trace)
    _synchronize(images.device)
    seconds = time.perf_counter() - started
    return AttackResult(
        adversarial=adversarial,
        kind=kind,
        epsilon=float(epsilon),
        alpha=float(alpha),
        steps=int(steps),
        random_start=bool(random_start),
        seconds=seconds,
        success_per_step=trace,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
