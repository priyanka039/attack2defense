"""Application factory. Run with ``uvicorn backend.app.main:create_app --factory --reload``."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from attack2defense import __version__
from attack2defense.utils import device_label, set_progress_enabled
from backend.app.db import Database
from backend.app.deps import Services
from backend.app.errors import install_error_handlers
from backend.app.routers import attacks, auth, datasets, evaluations, models, public, training
from backend.app.services.datasets import DatasetService
from backend.app.services.jobs import JobRunner
from backend.app.services.model_store import ModelStore
from backend.app.settings import DEFAULT_SECRET, Settings, get_settings

logger = logging.getLogger("attack2defense.api")


def build_services(settings: Settings) -> Services:
    database = Database(settings.database_url)
    database.fail_interrupted_jobs()
    return Services(
        settings=settings,
        database=database,
        datasets=DatasetService(settings),
        models=ModelStore(settings),
        jobs=JobRunner(database, inline=settings.inline_jobs),
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    services = build_services(settings)
    set_progress_enabled(False)
    if settings.secret_key == DEFAULT_SECRET:
        logger.warning("A2D_SECRET_KEY is not set. Tokens use a development key.")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        services.jobs.shutdown()

    app = FastAPI(
        title="Attack2Defense API",
        version=__version__,
        description="Generate FGSM and PGD attacks, train defended models, and evaluate robustness.",
        lifespan=lifespan,
    )
    app.state.services = services
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )
    install_error_handlers(app)

    api = APIRouter(prefix="/api")

    @api.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok", "version": __version__, "device": device_label(services.models.device)}

    for router in (auth, datasets, models, attacks, training, evaluations, public):
        api.include_router(router.router)
    app.include_router(api)
    return app
