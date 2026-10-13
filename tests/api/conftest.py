"""A test app with a tiny random-weight model and fake CIFAR-10 images. No downloads."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from torch.utils.data import TensorDataset

from attack2defense.config import CIFAR10_CLASSES
from attack2defense.models import build_pixel_model
from attack2defense.train import TrainConfig, build_run_config
from backend.app.main import create_app
from backend.app.settings import Settings

PASSWORD = "correct-horse-battery"


class FakeCifar(TensorDataset):
    def __init__(self, per_class: int = 5):
        count = per_class * len(CIFAR10_CLASSES)
        generator = torch.Generator().manual_seed(0)
        images = torch.rand(count, 3, 32, 32, generator=generator)
        labels = torch.arange(count) % len(CIFAR10_CLASSES)
        super().__init__(images, labels)
        self.targets = labels.tolist()


def save_model(folder: Path, mode: str = "clean") -> Path:
    torch.manual_seed(0)
    model = build_pixel_model("small_cnn").eval()
    config = build_run_config(TrainConfig(model="small_cnn", mode=mode))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    torch.save({"backbone": model.backbone.state_dict(), "config": config}, folder / "best.pt")
    return folder / "best.pt"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    save_model(tmp_path / "checkpoints" / "small_cnn_clean")

    # Results without weights, like a Colab run whose best.pt was not downloaded.
    run = tmp_path / "results" / "resnet18_pgd"
    (run / "test_eval").mkdir(parents=True)
    config = build_run_config(TrainConfig(model="resnet18", mode="pgd"))
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    pd.DataFrame([{"epoch": 1, "train_loss": 1.2, "val_clean": 0.5, "val_pgd": 0.3}]).to_csv(
        run / "history.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "model": "resnet18",
                "mode": "pgd",
                "clean_accuracy": 0.9,
                "fgsm_accuracy": 0.8,
                "pgd_accuracy": 0.79,
            }
        ]
    ).to_csv(run / "test_eval" / "summary.csv", index=False)
    return tmp_path


@pytest.fixture
def settings(workspace: Path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{(workspace / 'test.db').as_posix()}",
        secret_key="test-secret-key-that-is-long-enough-for-hs256",
        data_dir=workspace / "data",
        download_data=False,
        checkpoint_roots=[workspace / "checkpoints"],
        results_roots=[workspace / "results"],
        runs_dir=workspace / "runs",
        inline_jobs=True,
    )


@pytest.fixture
def app(settings: Settings):
    application = create_app(settings)
    application.state.services.datasets._cache["cifar10"] = FakeCifar()
    return application


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


def register(client: TestClient, email: str = "priya@example.com") -> dict:
    response = client.post("/api/auth/register", json={"email": email, "password": PASSWORD})
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def auth(client: TestClient) -> dict:
    return register(client)
