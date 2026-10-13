"""Certified-defense job: login required, fast fake smoother, public curve listing."""

from __future__ import annotations


def _fake_certify(model, loader, **kwargs):
    return {
        "sigma": kwargs.get("sigma", 0.25),
        "n0": kwargs["n0"],
        "n": kwargs["n"],
        "alpha": kwargs["alpha"],
        "images": 10,
        "naive_accuracy": 0.8,
        "rows": [{"radius": 0.0}],
        "curve": [
            {"radius": 0.0, "certified_accuracy": 0.8, "images": 10},
            {"radius": 0.25, "certified_accuracy": 0.4, "images": 10},
        ],
    }


def test_certify_requires_login(client):
    response = client.post("/api/certify", json={"model_id": "small_cnn_clean"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "not_authenticated"


def test_certify_job_returns_a_radius_curve(client, auth, monkeypatch):
    import attack2defense.smoothing as smoothing

    monkeypatch.setattr(smoothing, "certify_loader", _fake_certify)

    response = client.post(
        "/api/certify",
        headers=auth,
        json={"model_id": "small_cnn_clean", "n0": 20, "n": 40, "images_per_class": 1},
    )
    assert response.status_code == 202, response.text
    job = response.json()
    assert job["status"] == "succeeded", job
    assert job["result"]["kind"] == "certified"
    assert job["result"]["curve"][0]["radius"] == 0.0
    assert "rows" not in job["result"]
    listed = client.get("/api/certify", headers=auth).json()
    assert listed[0]["id"] == job["id"]


def test_public_results_include_certified_key(client):
    body = client.get("/api/public/results").json()
    assert "certified" in body
    assert "models" in body
