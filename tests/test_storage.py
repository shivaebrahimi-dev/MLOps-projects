import json

import pytest
import torch

from fashion_mlops import storage
from fashion_mlops.config import ServiceSettings
from fashion_mlops.model import LeNet5, deserialize_model, serialize_model


def test_bucket_name_uses_required_spelling(settings):
    assert settings.minio_bucket == "fashion-minst-bucket"


def test_upload_creates_bucket_and_versioned_key(fake_minio, settings):
    key = storage.upload_model(fake_minio, settings, "run-1", b"weights")
    assert key == "model/run-1/model.pt"
    assert fake_minio.buckets["fashion-minst-bucket"][key] == b"weights"


def test_later_runs_do_not_overwrite_earlier_models(fake_minio, settings):
    storage.upload_model(fake_minio, settings, "run-1", b"first")
    storage.upload_model(fake_minio, settings, "run-2", b"second")
    objects = fake_minio.buckets[settings.minio_bucket]
    assert objects["model/run-1/model.pt"] == b"first"
    assert objects["model/run-2/model.pt"] == b"second"


def test_latest_pointer_follows_most_recent_upload(fake_minio, settings):
    storage.upload_model(fake_minio, settings, "run-1", b"first")
    storage.upload_model(fake_minio, settings, "run-2", b"second", info={"val_accuracy": 0.9})
    pointer = json.loads(fake_minio.buckets[settings.minio_bucket]["model/latest.json"])
    assert pointer["run_id"] == "run-2"
    assert pointer["val_accuracy"] == 0.9
    assert storage.resolve_model_key(fake_minio, settings) == "model/run-2/model.pt"


def test_pinned_key_takes_priority_over_latest(fake_minio, settings):
    storage.upload_model(fake_minio, settings, "run-1", b"first")
    storage.upload_model(fake_minio, settings, "run-2", b"second")
    pinned = settings.model_copy(update={"model_object_key": "model/run-1/model.pt"})
    key = storage.resolve_model_key(fake_minio, pinned)
    assert storage.download_model(fake_minio, pinned, key) == b"first"


def test_no_model_yet_gives_clear_error(fake_minio, settings):
    with pytest.raises(storage.ModelNotFoundError, match="No trained model is available"):
        storage.resolve_model_key(fake_minio, settings)


def test_missing_object_gives_clear_error(fake_minio, settings):
    storage.ensure_bucket(fake_minio, settings)
    with pytest.raises(storage.ModelNotFoundError, match="model/nope/model.pt"):
        storage.download_model(fake_minio, settings, "model/nope/model.pt")


def test_unavailable_storage_raises_storage_error(fake_minio, settings):
    fake_minio.fail_uploads = True
    with pytest.raises(storage.StorageError, match="unavailable"):
        storage.upload_model(fake_minio, settings, "run-1", b"weights")
    assert "model/latest.json" not in fake_minio.buckets[settings.minio_bucket]


def test_missing_credentials_are_reported():
    empty = ServiceSettings(_env_file=None, minio_access_key="", minio_secret_key="")
    with pytest.raises(storage.StorageError, match="MINIO_ACCESS_KEY"):
        storage.make_client(empty)


def test_model_survives_serialisation_round_trip():
    model = LeNet5().eval()
    restored = deserialize_model(serialize_model(model))
    x = torch.randn(2, 1, 28, 28)
    assert torch.equal(model(x), restored(x))
