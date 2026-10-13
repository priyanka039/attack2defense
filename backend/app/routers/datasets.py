from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.app.deps import Services, get_services
from backend.app.services.imaging import to_data_url

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("")
def list_datasets(services: Services = Depends(get_services)) -> list[dict]:
    return services.datasets.list()


@router.get("/{name}")
def dataset_info(name: str, services: Services = Depends(get_services)) -> dict:
    return services.datasets.info(name)


@router.get("/{name}/samples")
def samples(
    name: str,
    label: str | None = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=24, ge=1, le=64),
    services: Services = Depends(get_services),
) -> dict:
    """Test-set images to attack, optionally filtered by class."""

    indices, total = services.datasets.samples(name, label, offset, limit)
    classes = services.datasets.info(name)["classes"]
    images, labels = services.datasets.items(name, indices) if indices else ([], [])
    return {
        "total": total,
        "offset": offset,
        "items": [
            {"index": index, "label": classes[int(target)], "image": to_data_url(image)}
            for index, image, target in zip(indices, images, labels, strict=True)
        ],
    }
