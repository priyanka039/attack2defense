from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from backend.app.db import User
from backend.app.deps import Services, get_services, optional_user
from backend.app.errors import ApiError

router = APIRouter(prefix="/models", tags=["models"])

_CONFIG_FIELDS = (
    "model",
    "mode",
    "epsilon",
    "pgd_steps",
    "adv_ratio",
    "epochs",
    "batch_size",
    "lr",
    "optimizer_name",
    "scheduler_name",
    "seed",
)


def _describe(entry, owned: bool) -> dict:
    return {**entry.to_dict(), "owner": "you" if owned else "public"}


@router.get("")
def list_models(
    user: User | None = Depends(optional_user), services: Services = Depends(get_services)
) -> list[dict]:
    """Public models, plus the caller's own training runs when logged in."""

    public = services.models.public_entries()
    own = services.models.user_entries(user.id) if user else {}
    return [_describe(entry, False) for entry in public.values()] + [
        _describe(entry, True) for entry in own.values()
    ]


@router.get("/{model_id}")
def model_detail(
    model_id: str, user: User | None = Depends(optional_user), services: Services = Depends(get_services)
) -> dict:
    entry = services.models.get(model_id, user.id if user else None)
    owned = user is not None and model_id in services.models.user_entries(user.id)
    return {
        **_describe(entry, owned),
        "config": {key: entry.config.get(key) for key in _CONFIG_FIELDS},
        "history": entry.history(),
        "has_examples": _examples(entry) is not None,
    }


@router.get("/{model_id}/examples.png", response_class=FileResponse)
def model_examples(
    model_id: str, user: User | None = Depends(optional_user), services: Services = Depends(get_services)
):
    """The before/after attack grid saved by the test evaluation."""

    path = _examples(services.models.get(model_id, user.id if user else None))
    if path is None:
        raise ApiError(404, "examples_not_found", f"'{model_id}' has no saved attack example grid.")
    return FileResponse(path, media_type="image/png")


def _examples(entry):
    if entry.run_dir is None:
        return None
    path = entry.run_dir / "test_eval" / "attack_examples.png"
    return path if path.exists() else None
