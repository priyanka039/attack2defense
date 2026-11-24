"""The production image serves the Vite build from the same origin as /api."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.settings import Settings


def test_spa_index_and_api_still_work(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>A2D</title>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")

    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'spa.db').as_posix()}",
        secret_key="test-secret-key-that-is-long-enough-for-hs256",
        data_dir=tmp_path / "data",
        download_data=False,
        checkpoint_roots=[],
        results_roots=[],
        runs_dir=tmp_path / "runs",
        frontend_dist=dist,
        inline_jobs=True,
    )
    client = TestClient(create_app(settings))

    assert client.get("/api/health").status_code == 200
    home = client.get("/")
    assert home.status_code == 200
    assert "A2D" in home.text
    # Client-side route: fall back to index.html
    guide = client.get("/guide")
    assert guide.status_code == 200
    assert "A2D" in guide.text
    assert client.get("/assets/app.js").status_code == 200
