from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends

from backend.app.deps import Services, get_services

router = APIRouter(prefix="/public", tags=["public"])


def _zeroshot(roots: list[Path]) -> list[dict]:
    """Last week's ImageNet ResNet-50 vs ViT-B/16 benchmark, if it was run."""

    paths = [
        path for root in map(Path, roots) if root.exists() for path in root.rglob("zeroshot_summary.csv")
    ]
    return pd.read_csv(sorted(paths)[0]).to_dict(orient="records") if paths else []


@router.get("/results")
def public_results(services: Services = Depends(get_services)) -> dict:
    """The reported experiment results. No login needed."""

    models = [entry.to_dict() for entry in services.models.public_entries().values() if entry.summary]
    certified = []
    for root in services.settings.results_roots:
        if not root.exists():
            continue
        for path in sorted(Path(root).rglob("certify.csv")):
            certified.append(
                {"source": path.parent.parent.name, "curve": pd.read_csv(path).to_dict(orient="records")}
            )
    return {"models": models, "zeroshot": _zeroshot(services.settings.results_roots), "certified": certified}
