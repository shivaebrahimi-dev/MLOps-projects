"""Model storage in MinIO.

Layout inside the bucket (``fashion-minst-bucket`` by default)::

    model/<mlflow-run-id>/model.pt   one object per training run, never overwritten
    model/latest.json                pointer to the most recently uploaded model
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any

import urllib3
from minio import Minio
from minio.error import S3Error

from fashion_mlops.config import ServiceSettings

MODEL_FILENAME = "model.pt"
LATEST_POINTER = "latest.json"


class StorageError(RuntimeError):
    """MinIO could not be used (unreachable, bad credentials, ...)."""


class ModelNotFoundError(StorageError):
    """The requested model object, pointer or bucket does not exist."""


def make_client(settings: ServiceSettings) -> Minio:
    if not settings.minio_access_key or not settings.minio_secret_key:
        raise StorageError(
            "MinIO credentials are not set. Define MINIO_ACCESS_KEY and MINIO_SECRET_KEY "
            "in the environment or in .env (see .env.example)."
        )
    # Short timeouts and few retries so an unavailable MinIO is reported in seconds, not minutes.
    http_client = urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=5, read=30),
        retries=urllib3.Retry(total=2, backoff_factor=0.5),
    )
    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
        http_client=http_client,
    )


def model_object_key(settings: ServiceSettings, run_id: str) -> str:
    return f"{settings.model_prefix}/{run_id}/{MODEL_FILENAME}"


def latest_pointer_key(settings: ServiceSettings) -> str:
    return f"{settings.model_prefix}/{LATEST_POINTER}"


def _storage_error(settings: ServiceSettings, action: str, exc: Exception) -> StorageError:
    return StorageError(
        f"Could not {action}: MinIO at {settings.minio_endpoint} is unavailable "
        f"or rejected the request ({exc})."
    )


def _put(client: Minio, bucket: str, key: str, data: bytes, content_type: str) -> None:
    client.put_object(bucket, key, io.BytesIO(data), length=len(data), content_type=content_type)


def _get(client: Minio, settings: ServiceSettings, key: str) -> bytes:
    bucket = settings.minio_bucket
    try:
        response = client.get_object(bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()
    except S3Error as exc:
        if exc.code in ("NoSuchKey", "NoSuchBucket"):
            raise ModelNotFoundError(f"'{key}' was not found in bucket '{bucket}'.") from exc
        raise _storage_error(settings, f"read '{key}'", exc) from exc
    except (urllib3.exceptions.HTTPError, OSError) as exc:
        raise _storage_error(settings, f"read '{key}'", exc) from exc


def ensure_bucket(client: Minio, settings: ServiceSettings) -> None:
    """Create the model bucket if it does not exist yet."""
    bucket = settings.minio_bucket
    try:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
    except (S3Error, urllib3.exceptions.HTTPError, OSError) as exc:
        raise _storage_error(settings, f"create or check bucket '{bucket}'", exc) from exc


def upload_model(
    client: Minio,
    settings: ServiceSettings,
    run_id: str,
    model_bytes: bytes,
    info: dict[str, Any] | None = None,
) -> str:
    """Upload a model under its run ID, then point ``latest.json`` at it. Returns the object key."""
    bucket = settings.minio_bucket
    key = model_object_key(settings, run_id)
    ensure_bucket(client, settings)
    try:
        _put(client, bucket, key, model_bytes, "application/octet-stream")
        # Written only after the model itself is stored, so the pointer never dangles.
        pointer = {
            "object_key": key,
            "run_id": run_id,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **(info or {}),
        }
        _put(
            client,
            bucket,
            latest_pointer_key(settings),
            json.dumps(pointer, indent=2).encode(),
            "application/json",
        )
    except (S3Error, urllib3.exceptions.HTTPError, OSError) as exc:
        raise _storage_error(settings, f"upload model to '{bucket}/{key}'", exc) from exc
    return key


def resolve_model_key(client: Minio, settings: ServiceSettings) -> str:
    """Which model should be served: the pinned ``MODEL_OBJECT_KEY``, else the latest upload."""
    if settings.model_object_key:
        return settings.model_object_key
    try:
        pointer = json.loads(_get(client, settings, latest_pointer_key(settings)))
    except ModelNotFoundError as exc:
        raise ModelNotFoundError(
            f"No trained model is available yet in bucket '{settings.minio_bucket}'. "
            "Run a training job first."
        ) from exc
    return pointer["object_key"]


def download_model(client: Minio, settings: ServiceSettings, key: str) -> bytes:
    return _get(client, settings, key)
