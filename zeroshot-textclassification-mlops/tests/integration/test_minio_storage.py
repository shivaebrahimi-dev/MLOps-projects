import dataclasses
import json
import time

import pytest

from app.services.minio_storage import PredictionStorage, create_client
from app.utils.errors import StorageError
from tests.helpers import make_result

pytestmark = pytest.mark.integration


def test_bucket_is_created_and_result_stored(minio_settings):
    storage = PredictionStorage.from_settings(minio_settings)
    client = create_client(minio_settings)
    assert not client.bucket_exists(minio_settings.minio_bucket)

    stored = storage.save(make_result())

    assert client.bucket_exists(minio_settings.minio_bucket)
    response = client.get_object(stored.bucket, stored.object_path)
    record = json.loads(response.read())
    assert record["request_id"] == make_result().request_id
    assert record["predicted_class"] == "sports"
    assert response.headers["Content-Type"] == "application/json"


def test_object_path_is_date_based(minio_settings):
    storage = PredictionStorage.from_settings(minio_settings)

    stored = storage.save(make_result())

    assert stored.object_path == f"predictions/2026/10/31/{make_result().request_id}.json"
    names = [
        item.object_name
        for item in create_client(minio_settings).list_objects(stored.bucket, recursive=True)
    ]
    assert names == [stored.object_path]


def test_second_save_uses_the_existing_bucket(minio_settings):
    storage = PredictionStorage.from_settings(minio_settings)

    storage.save(make_result(request_id="11111111-1111-4111-8111-111111111111"))
    storage.save(make_result(request_id="22222222-2222-4222-8222-222222222222"))

    objects = create_client(minio_settings).list_objects(
        minio_settings.minio_bucket, recursive=True
    )
    assert len(list(objects)) == 2


def test_unreachable_server_fails_quickly_with_a_clear_error(minio_settings):
    unreachable = dataclasses.replace(minio_settings, minio_endpoint="localhost:1")
    storage = PredictionStorage.from_settings(unreachable)

    started = time.perf_counter()
    with pytest.raises(StorageError, match="could not be reached at localhost:1"):
        storage.save(make_result())

    assert time.perf_counter() - started < 15


def test_wrong_credentials_give_a_clear_error_without_secrets(minio_settings):
    wrong = dataclasses.replace(minio_settings, minio_secret_key="wrong-secret-value")
    storage = PredictionStorage.from_settings(wrong)

    with pytest.raises(StorageError, match="refused the credentials") as error:
        storage.save(make_result())

    assert "wrong-secret-value" not in str(error.value)


def test_underscore_bucket_name_gives_a_clear_error(minio_settings):
    invalid = dataclasses.replace(minio_settings, minio_bucket="zeroshot_learning_output")
    storage = PredictionStorage.from_settings(invalid)

    with pytest.raises(StorageError, match="not a valid bucket name"):
        storage.save(make_result())
