import json
from unittest.mock import MagicMock

import httpx
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fashion_mlops import train
from fashion_mlops.config import TrainConfig
from fashion_mlops.storage import StorageError
from fashion_mlops.webhook import WebhookError, build_payload, send_webhook

PAYLOAD = build_payload(
    run_id="abc123",
    experiment="fashion-mnist-lenet5",
    bucket="fashion-minst-bucket",
    object_key="model/abc123/model.pt",
    val_accuracy=0.9,
    val_loss=0.3,
    epochs_run=5,
)


def test_payload_contains_run_and_model_references():
    assert PAYLOAD["status"] == "success"
    assert PAYLOAD["run_id"] == "abc123"
    assert PAYLOAD["model"] == {
        "bucket": "fashion-minst-bucket",
        "object_key": "model/abc123/model.pt",
        "uri": "s3://fashion-minst-bucket/model/abc123/model.pt",
    }
    assert PAYLOAD["metrics"]["val_accuracy"] == 0.9
    json.dumps(PAYLOAD)  # must be JSON-serialisable


def test_send_webhook_posts_json():
    received = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(json.loads(request.content))
        return httpx.Response(200)

    send_webhook("http://receiver.test/webhook", PAYLOAD, transport=httpx.MockTransport(handler))
    assert received == [PAYLOAD]


def test_send_webhook_raises_on_error_status():
    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    with pytest.raises(WebhookError):
        send_webhook("http://receiver.test/webhook", PAYLOAD, transport=transport)


# --- ordering: the webhook must only follow a successful upload -----------------------------


@pytest.fixture
def pipeline(monkeypatch, fake_minio):
    """run_training with MLflow, the dataset and the network replaced by local fakes."""
    data = TensorDataset(torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,)))
    loader = DataLoader(data, batch_size=16)
    fake_mlflow = MagicMock()
    fake_mlflow.start_run.return_value.__enter__.return_value.info.run_id = "run-1"
    sent = []

    monkeypatch.setattr(train, "mlflow", fake_mlflow)
    monkeypatch.setattr(train, "make_loaders", lambda config: (loader, loader))
    monkeypatch.setattr(train.storage, "make_client", lambda settings: fake_minio)
    monkeypatch.setattr(train, "send_webhook", lambda url, payload: sent.append((url, payload)))
    return sent


def test_webhook_sent_after_successful_upload(pipeline, fake_minio, settings):
    run_id = train.run_training(TrainConfig(epochs=1), settings)

    assert run_id == "run-1"
    assert "model/run-1/model.pt" in fake_minio.buckets[settings.minio_bucket]
    assert len(pipeline) == 1
    url, payload = pipeline[0]
    assert url == settings.webhook_url
    assert payload["model"]["object_key"] == "model/run-1/model.pt"


def test_no_webhook_when_upload_fails(pipeline, fake_minio, settings):
    fake_minio.fail_uploads = True
    with pytest.raises(StorageError):
        train.run_training(TrainConfig(epochs=1), settings)
    assert pipeline == []


def test_no_webhook_when_training_fails(pipeline, monkeypatch, settings):
    def broken_training(*args, **kwargs):
        raise RuntimeError("training blew up")

    monkeypatch.setattr(train, "train_model", broken_training)
    with pytest.raises(RuntimeError, match="training blew up"):
        train.run_training(TrainConfig(epochs=1), settings)
    assert pipeline == []


def test_webhook_skipped_when_url_not_configured(pipeline, settings):
    train.run_training(TrainConfig(epochs=1), settings.model_copy(update={"webhook_url": ""}))
    assert pipeline == []
