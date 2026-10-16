import io

import pytest
from minio.error import S3Error

from fashion_mlops.config import ServiceSettings


class FakeMinio:
    """In-memory stand-in for the MinIO client, covering only the calls the project makes."""

    def __init__(self) -> None:
        self.buckets: dict[str, dict[str, bytes]] = {}
        self.fail_uploads = False

    def bucket_exists(self, bucket):
        return bucket in self.buckets

    def make_bucket(self, bucket):
        self.buckets[bucket] = {}

    def put_object(self, bucket, key, data, length, content_type=None):
        if self.fail_uploads:
            raise OSError("connection refused")
        self.buckets[bucket][key] = data.read()

    def get_object(self, bucket, key):
        try:
            data = self.buckets[bucket][key]
        except KeyError:
            code = "NoSuchKey" if bucket in self.buckets else "NoSuchBucket"
            raise S3Error(None, code, "not found", key, "request-id", "host-id") from None
        response = io.BytesIO(data)
        response.release_conn = lambda: None
        return response


@pytest.fixture
def fake_minio():
    return FakeMinio()


@pytest.fixture
def settings():
    # _env_file=None keeps a developer's real .env out of the tests.
    return ServiceSettings(
        _env_file=None,
        minio_access_key="test",
        minio_secret_key="test-secret",
        mlflow_tracking_uri="memory://",
        webhook_url="http://receiver.test/webhook",
        model_object_key="",
    )
