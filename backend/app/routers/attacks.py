from __future__ import annotations

import torch
from fastapi import APIRouter, Depends

from attack2defense.attacks import run_attack
from backend.app.db import User
from backend.app.deps import Services, current_user, get_services
from backend.app.errors import ApiError
from backend.app.schemas import AttackItem, AttackRequest, AttackResponse, Prediction
from backend.app.services.imaging import from_data_url, heatmap_url, perturbation_url, to_data_url

router = APIRouter(prefix="/attacks", tags=["attacks"])

_TOP_K = 3


def _top(probabilities: torch.Tensor, classes: list[str]) -> list[Prediction]:
    values, indices = probabilities.topk(min(_TOP_K, len(classes)))
    return [
        Prediction(label=classes[int(index)], index=int(index), confidence=float(value))
        for value, index in zip(values, indices, strict=True)
    ]


def _inputs(request: AttackRequest, services: Services, image_size: int, classes: list[str]):
    """Images, attack labels, and known true labels (None for an unlabelled upload)."""

    if request.image is not None:
        images = from_data_url(request.image, image_size).unsqueeze(0)
        if request.label is None:
            return images, None, [None], [None]
        if request.label not in classes:
            raise ApiError(
                422, "invalid_label", f"'{request.label}' is not one of the model's classes.", "label"
            )
        index = classes.index(request.label)
        return images, torch.tensor([index]), [index], [None]

    indices = request.sample_indices or []
    if len(indices) > services.settings.max_attack_batch:
        raise ApiError(
            422,
            "batch_too_large",
            f"Attack at most {services.settings.max_attack_batch} images at a time.",
            "sample_indices",
        )
    images, labels = services.datasets.items(request.dataset, indices)
    if images.shape[-1] != image_size:
        images = torch.nn.functional.interpolate(
            images, size=image_size, mode="bilinear", align_corners=False
        )
    return images, labels, labels.tolist(), indices


def _attack(kind: str, request: AttackRequest, user: User, services: Services) -> AttackResponse:
    entry = services.models.get(request.model_id, user.id)
    model = services.models.load(entry)
    classes = list(entry.config.get("class_names", services.datasets.info(request.dataset)["classes"]))
    image_size = int(entry.config.get("image_size", 32))

    images, attack_labels, true_labels, sample_indices = _inputs(request, services, image_size, classes)
    device = services.models.device
    images = images.to(device)
    with torch.no_grad():
        clean_probabilities = model(images).softmax(dim=1)
    clean_predictions = clean_probabilities.argmax(dim=1)
    # An unlabelled upload is attacked away from whatever the model currently predicts.
    labels = clean_predictions if attack_labels is None else attack_labels.to(device)

    result = run_attack(
        model,
        images,
        labels,
        kind=kind,
        epsilon=request.epsilon,
        alpha=request.alpha,
        steps=request.steps,
        random_start=request.random_start,
    )
    with torch.no_grad():
        adversarial_probabilities = model(result.adversarial).softmax(dim=1)
    adversarial_predictions = adversarial_probabilities.argmax(dim=1)
    delta = result.adversarial - images

    items = []
    for position in range(images.shape[0]):
        true_label = true_labels[position]
        clean_top = _top(clean_probabilities[position], classes)
        adversarial_top = _top(adversarial_probabilities[position], classes)
        items.append(
            AttackItem(
                sample_index=sample_indices[position],
                true_label=classes[true_label] if true_label is not None else None,
                clean_image=to_data_url(images[position]),
                adversarial_image=to_data_url(result.adversarial[position]),
                perturbation_image=perturbation_url(delta[position], request.epsilon),
                heatmap_image=heatmap_url(delta[position], request.epsilon),
                clean_prediction=clean_top[0],
                adversarial_prediction=adversarial_top[0],
                clean_top=clean_top,
                adversarial_top=adversarial_top,
                fooled=bool(adversarial_predictions[position] != labels[position]),
                linf=float(delta[position].abs().max()),
                l2=float(delta[position].flatten().norm(p=2)),
            )
        )

    return AttackResponse(
        attack=kind,
        model_id=entry.id,
        epsilon=result.epsilon,
        alpha=result.alpha,
        steps=result.steps,
        random_start=result.random_start,
        seconds=result.seconds,
        success_per_step=result.success_per_step,
        success_rate=sum(item.fooled for item in items) / len(items),
        items=items,
    )


@router.post("/fgsm", response_model=AttackResponse)
def fgsm(
    request: AttackRequest, user: User = Depends(current_user), services: Services = Depends(get_services)
) -> AttackResponse:
    """One signed-gradient step of size epsilon (Goodfellow et al., 2015)."""

    return _attack("fgsm", request, user, services)


@router.post("/pgd", response_model=AttackResponse)
def pgd(
    request: AttackRequest, user: User = Depends(current_user), services: Services = Depends(get_services)
) -> AttackResponse:
    """Projected gradient descent inside the L-infinity ball (Madry et al., 2018)."""

    return _attack("pgd", request, user, services)
