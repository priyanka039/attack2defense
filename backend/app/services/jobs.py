"""Run long jobs (training, evaluation) in a background worker and record their progress."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable

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
            if job is None:
                return
            job.progress = max(0.0, min(1.0, fraction))
            job.message = message

    def add_history(self, row: dict) -> None:
        with self._database.session() as session:
            job = session.get(Job, self.job_id)
            if job is None:
                return
            job.history = [*job.history, row]


JobFunction = Callable[[JobContext], dict]


class JobRunner:
    """One dedicated worker thread draining a queue (reliable on small free hosts).

    ``ThreadPoolExecutor(max_workers=1)`` can wedge if a worker dies under memory
    pressure; a daemon loop that restarts itself keeps later jobs moving.
    """

    def __init__(self, database: Database, inline: bool = False):
        self._database = database
        self._inline = inline
        self._queue: queue.Queue[tuple[str, JobFunction, threading.Event] | None] = queue.Queue()
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        if not inline:
            self._ensure_worker()

    def submit(self, job_id: str, function: JobFunction) -> None:
        event = threading.Event()
        with self._lock:
            self._cancel_events[job_id] = event
        if self._inline:
            self._run(job_id, function, event)
            return
        self._ensure_worker()
        self._queue.put((job_id, function, event))
        logger.info("Queued job %s (pending=%s)", job_id, self._queue.qsize())

    def cancel(self, job_id: str) -> None:
        with self._lock:
            event = self._cancel_events.get(job_id)
        if event is not None:
            event.set()

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            for event in self._cancel_events.values():
                event.set()
        if not self._inline:
            self._queue.put(None)
            worker = self._worker
            if worker is not None and worker.is_alive():
                worker.join(timeout=2.0)

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stop.clear()
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="a2d-job-worker",
                daemon=True,
            )
            self._worker.start()
            logger.info("Started background job worker thread")

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                break
            job_id, function, event = item
            try:
                self._run(job_id, function, event)
            except Exception:
                logger.exception("Job worker crashed while handling %s", job_id)
                try:
                    self._finish(
                        job_id,
                        "failed",
                        message="Failed",
                        error="The background worker crashed. Start the job again.",
                    )
                except Exception:
                    logger.exception("Could not mark job %s as failed", job_id)
            finally:
                self._queue.task_done()

    def _finish(self, job_id: str, status: str, **fields) -> None:
        with self._database.session() as session:
            job = session.get(Job, job_id)
            if job is None:
                return
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
                if job is None:
                    logger.error("Job %s missing from the database when the worker started", job_id)
                    return
                job.status = "running"
                job.started_at = utcnow()
                job.message = "Starting — loading model and data (first run may download CIFAR-10)"
            logger.info("Running job %s", job_id)
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
