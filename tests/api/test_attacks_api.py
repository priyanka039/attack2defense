"""FGSM and PGD endpoints."""

from __future__ import annotations

import base64
import io

from PIL import Image


def test_pgd_returns_images_trace_and_cost(client, auth):
    epsilon = 8 / 255
    response = client.post(
        "/api/attacks/pgd",
        headers=auth,
        json={"model_id": "small_cnn_clean", "sample_indices": [0, 1, 2], "epsilon": epsilon, "steps": 4},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["attack"] == "pgd"
    assert body["steps"] == 4
    assert len(body["success_per_step"]) == 4
    assert body["seconds"] >= 0
    assert abs(body["alpha"] - 2.5 * epsilon / 4) < 1e-9
    assert len(body["items"]) == 3
    for item in body["items"]:
        assert item["linf"] <= epsilon + 1e-6
        assert item["true_label"] is not None
        for key in ("clean_image", "adversarial_image", "perturbation_image", "heatmap_image"):
            assert item[key].startswith("data:image/png;base64,")
        assert len(item["clean_top"]) == 3


def test_fgsm_is_a_single_step(client, auth):
    response = client.post(
        "/api/attacks/fgsm",
        headers=auth,
        json={
            "model_id": "small_cnn_clean",
            "sample_indices": [4],
            "epsilon": 2 / 255,
            "steps": 9,
            "alpha": 0.01,
        },
    )
    body = response.json()
    assert (body["steps"], body["random_start"], len(body["success_per_step"])) == (1, False, 1)
    assert abs(body["alpha"] - 2 / 255) < 1e-9


def test_uploaded_image_without_label(client, auth):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), color=(200, 30, 30)).save(buffer, format="PNG")
    upload = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    response = client.post(
        "/api/attacks/pgd", headers=auth, json={"model_id": "small_cnn_clean", "image": upload, "steps": 2}
    )
    assert response.status_code == 200, response.text
    item = response.json()["items"][0]
    assert item["sample_index"] is None
    assert item["true_label"] is None


def test_invalid_inputs_give_clear_messages(client, auth):
    cases = [
        (
            {"model_id": "small_cnn_clean", "sample_indices": [0], "epsilon": 0.5},
            "epsilon",
            "epsilon must be",
        ),
        ({"model_id": "small_cnn_clean", "sample_indices": [0], "steps": 0}, "steps", "steps:"),
        ({"model_id": "small_cnn_clean"}, None, "Give either sample_indices"),
        (
            {"model_id": "small_cnn_clean", "sample_indices": [99999]},
            "sample_indices",
            "outside the test set",
        ),
        ({"model_id": "small_cnn_clean", "image": "not-an-image"}, "image", "not a readable"),
    ]
    for payload, field, text in cases:
        response = client.post("/api/attacks/pgd", headers=auth, json=payload)
        assert response.status_code == 422, (payload, response.text)
        error = response.json()["error"]
        assert error["field"] == field, error
        assert text in error["message"], error


def test_missing_model_and_missing_weights(client, auth):
    missing = client.post("/api/attacks/pgd", headers=auth, json={"model_id": "ghost", "sample_indices": [0]})
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "model_not_found"

    no_weights = client.post(
        "/api/attacks/pgd", headers=auth, json={"model_id": "resnet18_pgd", "sample_indices": [0]}
    )
    assert no_weights.status_code == 409
    assert "best.pt" in no_weights.json()["error"]["message"]
