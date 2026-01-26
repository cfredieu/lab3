# tests/test_api.py
import pytest
from fastapi.testclient import TestClient


def test_health_endpoint(test_app):
    response = test_app.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "unhealthy")
    assert "model_loaded" in data
    # If model loaded successfully → should be healthy
    assert data["model_loaded"] is True, "Model failed to load in test environment"


def test_root_returns_html(test_app):
    response = test_app.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html" in response.text.lower() or "<!doctype" in response.text.lower()


def test_predict_endpoint_success(test_app, sample_image_bytes):
    files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    response = test_app.post("/predict", files=files)

    assert response.status_code == 200
    data = response.json()

    assert "predicted_class" in data
    assert "confidence" in data
    assert "top_5_predictions" in data
    assert len(data["top_5_predictions"]) == 5
    assert isinstance(data["confidence"], float)
    assert 0 <= data["confidence"] <= 1


def test_predict_rejects_non_image(test_app):
    files = {"file": ("evil.txt", b"not an image", "text/plain")}
    response = test_app.post("/predict", files=files)
    assert response.status_code == 400
    assert "File must be an image" in response.json()["detail"]


def test_predict_fails_when_model_not_loaded(monkeypatch, test_app):
    # Simulate model loading failure
    monkeypatch.setattr("src.retrieval.main.model_service", None)

    dummy_image = b"fake image bytes"
    files = {"file": ("fake.jpg", dummy_image, "image/jpeg")}
    response = test_app.post("/predict", files=files)

    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]
