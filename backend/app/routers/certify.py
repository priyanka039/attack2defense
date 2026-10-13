from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.db import User
from backend.app.deps import Services, current_user, get_services
from backend.app.errors import ApiError
from backend.app.routers.jobs_common import cancel_job, create_job, get_job, list_jobs
from backend.app.schemas import CertifyRequest, JobOut
from backend.app.services.tasks import run_certify_job

router = APIRouter(prefix="/certify", tags=["certify"])
KIND = "certify"


@router.post("", response_model=JobOut, status_code=202)
def start_certify(
    request: CertifyRequest, user: User = Depends(current_user), services: Services = Depends(get_services)
) -> JobOut:
    """Certified accuracy vs L2 radius (randomized smoothing). Separate from FGSM/PGD."""

    entry = services.models.get(request.model_id, user.id)
    if entry.checkpoint is None:
        raise ApiError(
            409,
            "model_has_no_weights",
            f"'{request.model_id}' has no best.pt. Download it into checkpoints/{request.model_id}/ first.",
            "model_id",
        )
    services.datasets.info(request.dataset)
    job = create_job(services, user, KIND, request.model_dump())
    services.jobs.submit(job.id, lambda ctx: run_certify_job(services, user.id, request, ctx))
    return get_job(services, user, KIND, job.id)


@router.get("", response_model=list[JobOut])
def certify_jobs(user: User = Depends(current_user), services: Services = Depends(get_services)):
    return list_jobs(services, user, KIND)


@router.get("/{job_id}", response_model=JobOut)
def certify_job(job_id: str, user: User = Depends(current_user), services: Services = Depends(get_services)):
    return get_job(services, user, KIND, job_id)


@router.delete("/{job_id}", response_model=JobOut)
def cancel_certify(
    job_id: str, user: User = Depends(current_user), services: Services = Depends(get_services)
):
    return cancel_job(services, user, KIND, job_id)
