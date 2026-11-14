from __future__ import annotations

from fastapi import APIRouter, Depends

from attack2defense.setup_check import inspect_setup
from attack2defense.utils import device_label
from backend.app.deps import Services, get_services
from backend.app.settings import PROJECT_ROOT

router = APIRouter(prefix="/setup", tags=["meta"])


@router.get("")
def setup_status(services: Services = Depends(get_services)) -> dict:
    """What is installed: dataset, weights, and results. No login needed."""

    settings = services.settings
    report = inspect_setup(
        settings.data_dir, settings.checkpoint_roots, settings.results_roots, base=PROJECT_ROOT
    )
    return {**report, "device": device_label(services.models.device)}
