import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from fashion_mlops import storage
from fashion_mlops.api.main import MAX_UPLOAD_BYTES, ModelService, app, get_model_service
from fashion_mlops.model import CLASS_NAMES, LeNet5, serialize_model


def png_bytes(size=(28, 28), mode="L") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color=128).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def service(fake_minio, settings):
    """A real ModelService backed by in-memory storage."""
    return ModelService(settings, client_factory=lambda s: fake_minio)


@pytest.fixture
def client(service):
    app.dependency_overrides[get_model_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def trained(fake_minio, settings):
    """Put an (untrained-weights) model in storage as run-1."""
    storage.upload_model(fake_minio, settings, "run-1", serialize_model(LeNet5()))


def post_image(client, data=None, content_type="image/png", filename="item.png"):
    data = png_bytes() if data is None else data
    return client.post("/predict", files={"file": (filename, data, content_type)})


def test_health_reports_missing_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_available"] is False
    assert "No trained model" in body["detail"]


def test_health_reports_loaded_model(client, trained):
    body = client.get("/health").json()
    assert body["model_available"] is True
    assert body["model_object_key"] == "model/run-1/model.pt"


def test_predict_returns_class_and_probabilities(client, trained):
    response = post_image(client)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in CLASS_NAMES
    assert body["predicted_class"] == CLASS_NAMES[body["class_index"]]
    assert set(body["probabilities"]) == set(CLASS_NAMES)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert body["confidence"] == pytest.approx(max(body["probabilities"].values()))
    assert body["model_object_key"] == "model/run-1/model.pt"


def test_predict_accepts_large_colour_images(client, trained):
    response = post_image(client, data=png_bytes(size=(300, 200), mode="RGB"))
    assert response.status_code == 200


def test_predict_without_model_returns_503(client):
    response = post_image(client)
    assert response.status_code == 503
    assert "No trained model" in response.json()["detail"]


def test_non_image_content_type_is_rejected(client, trained):
    response = post_image(client, data=b"hello", content_type="text/plain", filename="a.txt")
    assert response.status_code == 415


def test_corrupt_image_is_rejected(client, trained):
    response = post_image(client, data=b"this is not really a png")
    assert response.status_code == 400
    assert "could not be read as an image" in response.json()["detail"]


def test_empty_file_is_rejected(client, trained):
    assert post_image(client, data=b"").status_code == 400


def test_oversized_file_is_rejected(client, trained):
    response = post_image(client, data=b"0" * (MAX_UPLOAD_BYTES + 1))
    assert response.status_code == 413


def test_missing_file_field_is_rejected(client, trained):
    assert client.post("/predict").status_code == 422


def test_new_training_run_is_picked_up_without_restart(client, fake_minio, settings, trained):
    assert post_image(client).json()["model_object_key"] == "model/run-1/model.pt"
    storage.upload_model(fake_minio, settings, "run-2", serialize_model(LeNet5()))
    assert post_image(client).json()["model_object_key"] == "model/run-2/model.pt"


def test_cached_model_survives_storage_outage(service, fake_minio, trained):
    service.get()
    fake_minio.buckets.clear()
    _, key = service.get()
    assert key == "model/run-1/model.pt"
