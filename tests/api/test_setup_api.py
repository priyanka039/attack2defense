"""The setup report explains which runs can be attacked and which weights are missing."""

from __future__ import annotations

from attack2defense.setup_check import inspect_setup


def test_setup_report_is_public_and_lists_missing_weights(client):
    response = client.get("/api/setup")
    assert response.status_code == 200
    body = response.json()

    runs = {run["id"]: run for run in body["runs"]}
    assert runs["small_cnn_clean"]["has_weights"] is True
    assert runs["resnet18_pgd"]["has_weights"] is False
    assert runs["resnet18_pgd"]["has_results"] is True

    missing = {item["id"] for item in body["missing_reported_weights"]}
    assert missing == {"resnet18_clean", "resnet18_fgsm", "resnet18_pgd"}
    checks = {check["id"]: check for check in body["checks"]}
    assert checks["weights"]["ok"] is True
    assert checks["reported_weights"]["ok"] is False
    assert checks["reported_weights"]["hint"]
    assert body["device"]


def test_inspect_setup_with_nothing_installed(tmp_path):
    report = inspect_setup(
        tmp_path / "data", [tmp_path / "checkpoints"], [tmp_path / "results"], base=tmp_path
    )
    assert report["ready"] is False
    assert report["runs"] == []
    assert report["missing_reported_weights"][0]["expected_path"] == "checkpoints/resnet18_clean/best.pt"
