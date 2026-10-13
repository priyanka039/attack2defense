from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.db import User
from backend.app.deps import Services, current_user, get_services
from backend.app.errors import ApiError
from backend.app.routers.jobs_common import cancel_job, create_job, get_job, list_jobs
from backend.app.schemas import JobOut, TrainingRequest
from backend.app.services.tasks import run_training_job

router = APIRouter(prefix="/training/jobs", tags=["training"])
KIND = "training"


@router.post("", response_model=JobOut, status_code=202)
def start_training(
    request: TrainingRequest, user: User = Depends(current_user), services: Services = Depends(get_services)
) -> JobOut:
    """Queue a training run. Defended runs mix FGSM or PGD examples into each batch."""

    limits = services.settings
    if request.epochs > limits.max_train_epochs:
        raise ApiError(
            422, "too_many_epochs", f"Use at most {limits.max_train_epochs} epochs here.", "epochs"
        )
    if request.max_train_batches is None or request.max_train_batches > limits.max_train_batches:
        raise ApiError(
            422,
            "too_many_batches",
            f"Set max_train_batches to at most {limits.max_train_batches}. "
            "Full runs belong on a GPU (see README).",
            "max_train_batches",
        )
    services.datasets.info("cifar10")

    job = create_job(services, user, KIND, request.model_dump())
    services.jobs.submit(job.id, lambda ctx: run_training_job(services, user.id, job.id, request, ctx))
    return get_job(services, user, KIND, job.id)


@router.get("", response_model=list[JobOut])
def training_jobs(user: User = Depends(current_user), services: Services = Depends(get_services)):
    return list_jobs(services, user, KIND)


@router.get("/{job_id}", response_model=JobOut)
def training_job(job_id: str, user: User = Depends(current_user), services: Services = Depends(get_services)):
    return get_job(services, user, KIND, job_id)


@router.delete("/{job_id}", response_model=JobOut)
def cancel_training(
    job_id: str, user: User = Depends(current_user), services: Services = Depends(get_services)
):
    return cancel_job(services, user, KIND, job_id)
