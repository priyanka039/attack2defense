"""Create, list, read, and cancel a user's jobs. Shared by the job routers."""

from __future__ import annotations

from sqlalchemy import select

from backend.app.db import ACTIVE_STATUSES, Job, User
from backend.app.deps import Services
from backend.app.errors import not_found
from backend.app.schemas import JobOut


def create_job(services: Services, user: User, kind: str, params: dict) -> Job:
    with services.database.session() as session:
        job = Job(user_id=user.id, kind=kind, params=params)
        session.add(job)
        session.flush()
        return job


def list_jobs(services: Services, user: User, kind: str) -> list[JobOut]:
    with services.database.session() as session:
        jobs = session.scalars(
            select(Job).where(Job.user_id == user.id, Job.kind == kind).order_by(Job.created_at.desc())
        ).all()
        return [JobOut.model_validate(job) for job in jobs]


def get_job(services: Services, user: User, kind: str, job_id: str) -> JobOut:
    with services.database.session() as session:
        job = session.get(Job, job_id)
        # Another user's job is reported as missing, not forbidden, so ids do not leak.
        if job is None or job.user_id != user.id or job.kind != kind:
            raise not_found(f"{kind}_job", job_id)
        return JobOut.model_validate(job)


def cancel_job(services: Services, user: User, kind: str, job_id: str) -> JobOut:
    job = get_job(services, user, kind, job_id)
    if job.status in ACTIVE_STATUSES:
        with services.database.session() as session:
            stored = session.get(Job, job_id)
            if stored.status == "queued":
                stored.status = "cancelled"
                stored.message = "Cancelled before it started"
            else:
                stored.status = "cancelling"
                stored.message = "Stopping after the current batch"
        services.jobs.cancel(job_id)
    return get_job(services, user, kind, job_id)
