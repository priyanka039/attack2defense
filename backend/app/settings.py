"""Runtime configuration, read from ``A2D_*`` environment variables or ``backend/.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SECRET = "development-only-secret-change-me-in-backend-env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="A2D_", env_file=PROJECT_ROOT / "backend" / ".env")

    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'backend' / 'attack2defense.db').as_posix()}"
    secret_key: str = DEFAULT_SECRET
    token_minutes: int = 12 * 60

    data_dir: Path = PROJECT_ROOT / "data"
    download_data: bool = True
    checkpoint_roots: list[Path] = [PROJECT_ROOT / "checkpoints", PROJECT_ROOT / "outputs" / "cpu_preview"]
    results_roots: list[Path] = [PROJECT_ROOT / "results"]
    runs_dir: Path = PROJECT_ROOT / "outputs" / "api_runs"

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # Limits that keep a CPU-only server responsive.
    max_attack_batch: int = 16
    max_eval_images_per_class: int = 200
    max_train_epochs: int = 30
    max_train_batches: int = 400

    # Run jobs in the request thread. Used by the tests.
    inline_jobs: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
