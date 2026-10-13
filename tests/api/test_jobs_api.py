"""Training and evaluation jobs: lifecycle, ownership, export, and cancellation."""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

from attack2defense.train import TrainingCancelled
from backend.app.db import Database, Job, User
from backend.app.services import tasks
from backend.app.services.jobs import JobRunner
from tests.api.conftest import register, save_model


def _fake_training(config, hooks):
    """Stands in for run_training: reports progress and writes a checkpoint."""

    for epoch in range(1, config.epochs + 1):
        hooks.on_batch(epoch, 1, 1)
        hooks.on_epoch({"epoch": epoch, "train_loss": 1.0 / epoch, "val_clean": 0.5, "val_pgd": 0.2})
    checkpoint = save_model(Path(config.output_dir), mode=config.mode)
    return {"checkpoint": str(checkpoint)}


@pytest.fixture
def fake_training(monkeypatch):
    monkeypatch.setattr(tasks, "run_training", _fake_training)


def test_training_job_lifecycle_and_ownership(client, auth, fake_training):
    response = client.post(
        "/api/training/jobs",
        headers=auth,
        json={"mode": "pgd", "epochs": 2, "adv_ratio": 0.5, "max_train_batches": 5},
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]

    job = client.get(f"/api/training/jobs/{job_id}", headers=auth).json()
    assert job["status"] == "succeeded", job
    assert job["progress"] == 1.0
    assert [row["epoch"] for row in job["history"]] == [1, 2]
    assert job["params"]["adv_ratio"] == 0.5
    model_id = job["result"]["model_id"]
    assert model_id == f"job_{job_id}"
    assert "pgd_accuracy" in job["result"]["summary"]

    mine = {model["id"]: model for model in client.get("/api/models", headers=auth).json()}
    assert mine[model_id]["owner"] == "you"
    assert [job["id"] for job in client.get("/api/training/jobs", headers=auth).json()] == [job_id]

    other = register(client, "someone@example.com")
    assert client.get(f"/api/training/jobs/{job_id}", headers=other).status_code == 404
    assert model_id not in {model["id"] for model in client.get("/api/models", headers=other).json()}
    assert client.get("/api/training/jobs", headers=other).json() == []


def test_training_limits(client, auth):
    too_long = client.post("/api/training/jobs", headers=auth, json={"epochs": 500, "max_train_batches": 5})
    assert too_long.status_code == 422
    assert too_long.json()["error"]["field"] == "epochs"

    full_run = client.post("/api/training/jobs", headers=auth, json={"max_train_batches": None})
    assert full_run.status_code == 422
    assert full_run.json()["error"]["field"] == "max_train_batches"


def test_evaluation_sweep_flags_and_export(client, auth):
    response = client.post(
        "/api/evaluations",
        headers=auth,
        json={
            "model_ids": ["small_cnn_clean"],
            "epsilons": [8 / 255, 2 / 255, 4 / 255],
            "threshold": 0.95,
            "images_per_class": 2,
            "pgd_steps": 2,
        },
    )
    assert response.status_code == 202, response.text
    job = client.get(f"/api/evaluations/{response.json()['id']}", headers=auth).json()
    assert job["status"] == "succeeded", job
    result = job["result"]
    assert result["images"] == 20
    rows = result["models"][0]["rows"]
    assert [row["epsilon"] for row in rows] == sorted([2 / 255, 4 / 255, 8 / 255])
    assert all(row["pgd_below_threshold"] for row in rows)
    assert len(result["flags"]) >= 3

    csv = client.get(f"/api/evaluations/{job['id']}/export", params={"format": "csv"}, headers=auth)
    assert csv.headers["content-type"].startswith("text/csv")
    assert "attachment" in csv.headers["content-disposition"]
    frame = pd.read_csv(io.StringIO(csv.text))
    assert len(frame) == 3
    assert {"model_id", "clean_accuracy", "pgd_accuracy", "pgd_below_threshold"} <= set(frame.columns)

    exported = client.get(
        f"/api/evaluations/{job['id']}/export", params={"format": "json"}, headers=auth
    ).json()
    assert exported["threshold"] == 0.95


def test_evaluation_rejects_models_without_weights(client, auth):
    response = client.post("/api/evaluations", headers=auth, json={"model_ids": ["resnet18_pgd"]})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "model_has_no_weights"


def test_background_job_can_be_cancelled(tmp_path):
    database = Database(f"sqlite:///{(tmp_path / 'jobs.db').as_posix()}")
    with database.session() as session:
        session.add(User(id=1, email="a@example.com", password_hash="x"))
        session.add(Job(id="job1", user_id=1, kind="training"))
    runner = JobRunner(database)
    started = threading.Event()

    def work(ctx):
        started.set()
        while not ctx.cancelled():
            time.sleep(0.01)
        raise TrainingCancelled

    runner.submit("job1", work)
    assert started.wait(5)
    runner.cancel("job1")
    for _ in range(200):
        with database.session() as session:
            status = session.get(Job, "job1").status
        if status == "cancelled":
            break
        time.sleep(0.02)
    assert status == "cancelled"
    runner.shutdown()
