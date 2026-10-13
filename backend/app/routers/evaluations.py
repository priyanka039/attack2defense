from __future__ import annotations

import io
import json
from typing import Literal

import pandas as pd
from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from backend.app.db import User
from backend.app.deps import Services, current_user, get_services
from backend.app.errors import ApiError
from backend.app.routers.jobs_common import cancel_job, create_job, get_job, list_jobs
from backend.app.schemas import EvaluationRequest, JobOut
from backend.app.services.tasks import run_evaluation_job

router = APIRouter(prefix="/evaluations", tags=["evaluations"])
KIND = "evaluation"


@router.post("", response_model=JobOut, status_code=202)
def start_evaluation(
    request: EvaluationRequest, user: User = Depends(current_user), services: Services = Depends(get_services)
) -> JobOut:
    """Clean, FGSM, and PGD accuracy at each epsilon, for every model, on the same test images."""

    limit = services.settings.max_eval_images_per_class
    if request.images_per_class > limit:
        raise ApiError(422, "too_many_images", f"Use at most {limit} images per class.", "images_per_class")
    services.datasets.info(request.dataset)
    for model_id in request.model_ids:
        entry = services.models.get(model_id, user.id)
        if entry.checkpoint is None:
            raise ApiError(
                409,
                "model_has_no_weights",
                f"'{model_id}' has results but no best.pt. Download it into checkpoints/{model_id}/ first.",
                "model_ids",
            )

    job = create_job(services, user, KIND, request.model_dump())
    services.jobs.submit(job.id, lambda ctx: run_evaluation_job(services, user.id, request, ctx))
    return get_job(services, user, KIND, job.id)


@router.get("", response_model=list[JobOut])
def evaluations(user: User = Depends(current_user), services: Services = Depends(get_services)):
    return list_jobs(services, user, KIND)


@router.get("/{job_id}", response_model=JobOut)
def evaluation(job_id: str, user: User = Depends(current_user), services: Services = Depends(get_services)):
    return get_job(services, user, KIND, job_id)


@router.delete("/{job_id}", response_model=JobOut)
def cancel_evaluation(
    job_id: str, user: User = Depends(current_user), services: Services = Depends(get_services)
):
    return cancel_job(services, user, KIND, job_id)


@router.get("/{job_id}/export")
def export_evaluation(
    job_id: str,
    format: Literal["csv", "json"] = Query(default="csv"),
    user: User = Depends(current_user),
    services: Services = Depends(get_services),
) -> Response:
    """Download a finished evaluation as CSV (one row per model and epsilon) or full JSON."""

    job = get_job(services, user, KIND, job_id)
    if job.status != "succeeded" or job.result is None:
        raise ApiError(
            409, "evaluation_not_finished", f"This evaluation is {job.status}; export it once it finishes."
        )
    filename = f"evaluation_{job_id[:8]}.{format}"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    if format == "json":
        return Response(json.dumps(job.result, indent=2), media_type="application/json", headers=headers)

    rows = [
        {"model_id": model["model_id"], "label": model["label"], "tag": model["tag"]}
        | {key: value for key, value in row.items() if key != "per_class"}
        for model in job.result["models"]
        for row in model["rows"]
    ]
    buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    return Response(buffer.getvalue(), media_type="text/csv", headers=headers)
