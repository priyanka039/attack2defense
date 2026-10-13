"""Login, protected routes, error shape, and the public catalogue."""

from __future__ import annotations

from tests.api.conftest import PASSWORD, register


def _error(response) -> dict:
    body = response.json()
    assert set(body) == {"error"}, body
    assert set(body["error"]) == {"code", "message", "field"}
    return body["error"]


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["device"]


def test_register_login_and_me(client):
    headers = register(client, "Priya@Example.com")
    me = client.get("/api/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["email"] == "priya@example.com"

    login = client.post("/api/auth/login", json={"email": "priya@example.com", "password": PASSWORD})
    assert login.status_code == 200
    assert login.json()["token_type"] == "bearer"
    assert login.json()["expires_in"] > 0


def test_duplicate_email_and_wrong_password(client):
    register(client)
    duplicate = client.post("/api/auth/register", json={"email": "priya@example.com", "password": PASSWORD})
    assert duplicate.status_code == 409
    assert _error(duplicate)["code"] == "email_taken"

    wrong = client.post(
        "/api/auth/login", json={"email": "priya@example.com", "password": "not-the-password"}
    )
    assert wrong.status_code == 401
    assert _error(wrong)["message"] == "Wrong email or password."


def test_short_password_is_a_readable_validation_error(client):
    response = client.post("/api/auth/register", json={"email": "a@example.com", "password": "short"})
    assert response.status_code == 422
    error = _error(response)
    assert error["field"] == "password"
    assert error["message"].startswith("password:")


def test_run_creation_needs_login(client):
    for path in ("/api/attacks/pgd", "/api/attacks/fgsm", "/api/training/jobs", "/api/evaluations"):
        response = client.post(path, json={})
        assert response.status_code == 401, path
        assert _error(response)["code"] == "not_authenticated"


def test_bad_token_is_rejected(client):
    response = client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-token"})
    assert response.status_code == 401
    assert _error(response)["code"] == "invalid_token"


def test_public_results_need_no_login(client):
    body = client.get("/api/public/results").json()
    ids = [model["id"] for model in body["models"]]
    assert "resnet18_pgd" in ids
    pgd = next(model for model in body["models"] if model["id"] == "resnet18_pgd")
    assert pgd["summary"]["pgd_accuracy"] == 0.79
    assert pgd["tag"] == "pgd5_eps2_ratio1"
    assert pgd["has_weights"] is False


def test_model_catalogue_and_detail(client):
    models = {model["id"]: model for model in client.get("/api/models").json()}
    assert models["small_cnn_clean"]["has_weights"] is True
    assert models["small_cnn_clean"]["owner"] == "public"

    detail = client.get("/api/models/resnet18_pgd").json()
    assert detail["history"][0]["val_pgd"] == 0.3
    assert detail["config"]["mode"] == "pgd"

    missing = client.get("/api/models/nope")
    assert missing.status_code == 404
    assert _error(missing)["code"] == "model_not_found"


def test_dataset_samples_filter_by_class(client):
    datasets = client.get("/api/datasets").json()
    assert datasets[0]["name"] == "cifar10"

    body = client.get("/api/datasets/cifar10/samples", params={"label": "cat", "limit": 3}).json()
    assert body["total"] == 5
    assert len(body["items"]) == 3
    assert all(item["label"] == "cat" for item in body["items"])
    assert body["items"][0]["image"].startswith("data:image/png;base64,")

    bad = client.get("/api/datasets/cifar10/samples", params={"label": "unicorn"})
    assert bad.status_code == 422
    assert _error(bad)["code"] == "invalid_label"
