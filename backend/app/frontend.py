"""Serve the Vite production build from the same origin as the API."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.settings import PROJECT_ROOT

# Paths FastAPI already owns; the SPA catch-all must not steal them.
_RESERVED = frozenset({"api", "docs", "redoc", "openapi.json", "assets"})


def install_frontend(app: FastAPI, dist: Path | None = None) -> None:
    """Mount ``frontend/dist`` when it exists (production / Render image)."""

    root = dist or (PROJECT_ROOT / "frontend" / "dist")
    if not root.is_dir() or not (root / "index.html").is_file():
        return

    assets = root / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

    @app.get("/")
    async def spa_index() -> FileResponse:
        return FileResponse(root / "index.html")

    @app.get("/{full_path:path}")
    async def spa(full_path: str) -> FileResponse:
        first = full_path.split("/", 1)[0]
        if first in _RESERVED:
            raise HTTPException(status_code=404, detail="Not found")
        candidate = root / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(root / "index.html")
