import json
import re

import pytest
import urllib3
from minio.error import S3Error

from app.services.minio_storage import PredictionStorage, build_record, object_path
from app.utils.config import PROJECT_ROOT
from app.utils.errors import StorageError
from tests.helpers import make_result

SCHEMA = json.loads(
    (
        PROJECT_ROOT
        / "specs"
        / "001-zero-shot-platform"
        / "contracts"
        / "prediction-record.schema.json"
    ).read_text(encoding="utf-8")
)


def s3_error(code):
    return S3Error(None, code, "message", "resource", "request", "host")


class FakeClient:
    def __init__(self, exists=True, failure=None, make_failure=None):
        self.exists = exists
        self.failure = failure
        self.make_failure = make_failure
        self.made = []
        self.objects = {}

    def bucket_exists(self, bucket):
        if self.failure:
            raise self.failure
        return self.exists

    def make_bucket(self, bucket):
        if self.make_failure:
            raise self.make_failure
        self.made.append(bucket)

    def put_object(self, bucket, path, data, length, content_type):
        self.objects[(bucket, path)] = {
            "body": data.read(),
            "length": length,
            "content_type": content_type,
        }


def storage_with(client):
    return PredictionStorage(client, "test-bucket", "0.1.0", "localhost:9000")


def stored_body(client):
    return json.loads(next(iter(client.objects.values()))["body"])


def test_bucket_is_created_when_missing():
    client = FakeClient(exists=False)

    storage_with(client).save(make_result())

    assert client.made == ["test-bucket"]


def test_existing_bucket_is_not_created_again():
    client = FakeClient(exists=True)

    storage_with(client).save(make_result())

    assert client.made == []


def test_bucket_created_by_another_session_is_tolerated():
    client = FakeClient(exists=False, make_failure=s3_error("BucketAlreadyOwnedByYou"))

    stored = storage_with(client).save(make_result())

    assert stored.bucket == "test-bucket"


def test_object_path_uses_the_required_date_format():
    assert (
        object_path(make_result())
        == "predictions/2026/10/31/6a6d6a85-9dbb-4f3e-9f89-5d8e4e8c1b2a.json"
    )


def test_path_uses_zero_padded_dates():
    assert object_path(make_result(timestamp="2026-01-05T00:00:00Z")).startswith(
        "predictions/2026/01/05/"
    )


def test_result_is_stored_as_valid_json_at_the_returned_path():
    client = FakeClient()

    stored = storage_with(client).save(make_result())

    saved = client.objects[("test-bucket", stored.object_path)]
    assert saved["content_type"] == "application/json"
    assert saved["length"] == len(saved["body"])
    assert json.loads(saved["body"])["request_id"] == make_result().request_id


def test_record_has_every_required_field_and_no_unknown_ones():
    record = build_record(make_result(), "0.1.0", session_id="session")

    assert set(SCHEMA["required"]) <= set(record)
    assert set(record) <= set(SCHEMA["properties"])


def test_record_follows_the_schema_patterns():
    record = build_record(make_result(), "0.1.0")

    assert re.match(SCHEMA["properties"]["timestamp"]["pattern"], record["timestamp"])
    assert record["ranked_predictions"][0] == {"label": "sports", "score": 0.96}
    assert record["predicted_class"] == record["ranked_predictions"][0]["label"]
    assert record["confidence"] == record["ranked_predictions"][0]["score"]
    assert record["application_version"] == "0.1.0"
    assert record["model"] == "org/model"
    assert record["model_name"] == "Test Model"
    assert record["model_revision"] == "abc123"


def test_non_ascii_text_is_stored_intact():
    client = FakeClient()
    result = make_result()
    object.__setattr__(result, "input_text", "L'équipe a gagné")

    storage_with(client).save(result)

    assert stored_body(client)["input_text"] == "L'équipe a gagné"


def test_session_id_is_stored():
    client = FakeClient()

    storage_with(client).save(make_result(), session_id="session-1")

    assert stored_body(client)["session_id"] == "session-1"


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (urllib3.exceptions.MaxRetryError(None, "url"), "could not be reached at localhost:9000"),
        (OSError("connection refused"), "could not be reached"),
        (s3_error("InvalidAccessKeyId"), "refused the credentials"),
        (s3_error("SignatureDoesNotMatch"), "refused the credentials"),
        (s3_error("InternalError"), "error InternalError"),
        (ValueError("invalid bucket name"), "not a valid bucket name"),
        (s3_error("InvalidBucketName"), "not a valid bucket name"),
    ],
)
def test_storage_errors_are_clear(failure, message):
    client = FakeClient(failure=failure)

    with pytest.raises(StorageError, match=message) as error:
        storage_with(client).save(make_result())

    assert "not saved" in error.value.user_message
    assert client.objects == {}
