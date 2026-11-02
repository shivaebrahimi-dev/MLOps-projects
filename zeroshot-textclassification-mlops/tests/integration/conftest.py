import os
import uuid

import pytest
import urllib3

from app.services.minio_storage import create_client
from app.utils.config import parse_settings


def _settings(bucket):
    return parse_settings(
        {
            "MINIO_ENDPOINT": os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
            "MINIO_ACCESS_KEY": os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
            "MINIO_SECRET_KEY": os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
            "MINIO_BUCKET": bucket,
        }
    )


def _minio_is_reachable(settings):
    try:
        urllib3.PoolManager(retries=False, timeout=2).request(
            "GET", f"http://{settings.minio_endpoint}/minio/health/live"
        )
    except urllib3.exceptions.HTTPError:
        return False
    return True


@pytest.fixture
def minio_settings():
    """Settings for a unique bucket, removed after the test. CI requires MinIO to be running."""
    settings = _settings(f"zeroshot-test-{uuid.uuid4().hex[:12]}")
    if not _minio_is_reachable(settings):
        if os.environ.get("REQUIRE_MINIO", "").lower() == "true":
            pytest.fail(f"MinIO is required but not reachable at {settings.minio_endpoint}.")
        pytest.skip(f"MinIO is not reachable at {settings.minio_endpoint}.")

    yield settings

    client = create_client(settings)
    if client.bucket_exists(settings.minio_bucket):
        for item in client.list_objects(settings.minio_bucket, recursive=True):
            client.remove_object(settings.minio_bucket, item.object_name)
        client.remove_bucket(settings.minio_bucket)
