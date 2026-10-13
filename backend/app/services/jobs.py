"""Run long jobs (training, evaluation) in a background worker and record their progress."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from attack2defense.train import TrainingCancelled
from backend.app.db import Database, Job, utcnow
from backend.app.errors import ApiError

logger = logging.getLogger("attack2defense.jobs")

_PROGRESS_INTERVAL = 0.5


class JobContext:
    """Handed to a job function so it can report progress and notice a cancel."""

    def __init__(self, job_id: str, database: Database, cancel_event: threading.Event):
        self.job_id = job_id
        self._database = database
        self._cancel = cancel_event
        self._last_update = 0.0

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def progress(self, fraction: float, message: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_update < _PROGRESS_INTERVAL:
            return
        self._last_update = now
        with self._database.session() as session:
            job = session.get(Job, self.job_id)
            job.progress = max(0.0, min(1.0, fraction))
            job.message = message

    def add_history(self, row: dict) -> None:
        with self._database.session() as session:
            job = session.get(Job, self.job_id)
            job.history = [*job.history, row]


JobFunction = Callable[[JobContext], dict]


class JobRunner:
    """One worker thread: training and evaluation are CPU or GPU bound, so they queue."""

    def __init__(self, database: Database, inline: bool = False):
        self._database = database
        self._inline = inline
        self._executor = None if inline else ThreadPoolExecutor(max_workers=1, thread_name_prefix="job")
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def submit(self, job_id: str, function: JobFunction) -> None:
        event = threading.Event()
        with self._lock:
            self._cancel_events[job_id] = event
        if self._executor is None:
            self._run(job_id, function, event)
        else:
            self._executor.submit(self._run, job_id, function, event)

    def cancel(self, job_id: str) -> None:
        with self._lock:
            event = self._cancel_events.get(job_id)
        if event is not None:
            event.set()

    def shutdown(self) -> None:
        with self._lock:
            for event in self._cancel_events.values():
                event.set()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _finish(self, job_id: str, status: str, **fields) -> None:
        with self._database.session() as session:
            job = session.get(Job, job_id)
            job.status = status
            job.finished_at = utcnow()
            for name, value in fields.items():
                setattr(job, name, value)

    def _run(self, job_id: str, function: JobFunction, event: threading.Event) -> None:
        try:
            if event.is_set():
                self._finish(job_id, "cancelled", message="Cancelled before it started")
                return
            with self._database.session() as session:
                job = session.get(Job, job_id)
                job.status = "running"
                job.started_at = utcnow()
                job.message = "Starting"
            result = function(JobContext(job_id, self._database, event))
            self._finish(job_id, "succeeded", progress=1.0, message="Finished", result=result)
        except TrainingCancelled:
            self._finish(job_id, "cancelled", message="Cancelled")
        except (ApiError, ValueError, RuntimeError) as exc:
            message = exc.message if isinstance(exc, ApiError) else str(exc)
            self._finish(job_id, "failed", message="Failed", error=message)
        except Exception as exc:
            logger.exception("Job %s crashed", job_id)
            self._finish(job_id, "failed", message="Failed", error=f"Unexpected error: {exc}")
        finally:
            with self._lock:
                self._cancel_events.pop(job_id, None)
