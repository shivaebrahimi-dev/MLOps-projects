"""Saves prediction records to MinIO, the system of record for classification outputs."""

import io
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import urllib3
from minio import Minio
from minio.error import MinioException, S3Error

from app.services.classifier import TIMESTAMP_FORMAT, ClassificationResult
from app.utils.config import Settings
from app.utils.errors import StorageError
from app.utils.logging_config import log_event

logger = logging.getLogger(__name__)

_ALREADY_EXISTS_CODES = {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}
_AUTHENTICATION_CODES = {"AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch"}


@dataclass(frozen=True)
class StoredPrediction:
    bucket: str
    object_path: str


def object_path(result: ClassificationResult) -> str:
    moment = datetime.strptime(result.timestamp, TIMESTAMP_FORMAT)
    return f"predictions/{moment:%Y/%m/%d}/{result.request_id}.json"


def build_record(
    result: ClassificationResult, application_version: str, session_id: str | None = None
) -> dict[str, Any]:
    return {
        "request_id": result.request_id,
        "timestamp": result.timestamp,
        "input_text": result.input_text,
        "predicted_class": result.predicted_class,
        "confidence": result.confidence,
        "candidate_labels": list(result.candidate_labels),
        "ranked_predictions": [
            {"label": item.label, "score": item.score} for item in result.ranked_predictions
        ],
        "model_name": result.model_name,
        "model": result.model_id,
        "model_revision": result.model_revision,
        "application_version": application_version,
        "session_id": session_id,
        "latency_ms": round(result.latency_ms, 1),
    }


def create_client(settings: Settings) -> Minio:
    # The client's defaults retry for a long time, which would leave the page waiting.
    http_client = urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=3, read=10),
        retries=urllib3.Retry(total=1, backoff_factor=0.2),
    )
    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
        http_client=http_client,
    )


class PredictionStorage:
    def __init__(self, client: Any, bucket: str, application_version: str, endpoint: str) -> None:
        self._client = client
        self.bucket = bucket
        self._application_version = application_version
        self._endpoint = endpoint

    @classmethod
    def from_settings(cls, settings: Settings) -> "PredictionStorage":
        return cls(
            create_client(settings),
            settings.minio_bucket,
            settings.app_version,
            settings.minio_endpoint,
        )

    def save(self, result: ClassificationResult, session_id: str | None = None) -> StoredPrediction:
        path = object_path(result)
        data = json.dumps(
            build_record(result, self._application_version, session_id), ensure_ascii=False
        ).encode("utf-8")
        try:
            self._ensure_bucket()
            self._client.put_object(
                self.bucket, path, io.BytesIO(data), len(data), content_type="application/json"
            )
        except (MinioException, urllib3.exceptions.HTTPError, OSError, ValueError) as error:
            log_event(
                logger,
                "prediction_store_failed",
                level=logging.ERROR,
                request_id=result.request_id,
                bucket=self.bucket,
                error_type=type(error).__name__,
            )
            raise StorageError(self._failure_message(error)) from None

        log_event(
            logger,
            "prediction_stored",
            request_id=result.request_id,
            bucket=self.bucket,
            object_path=path,
            size_bytes=len(data),
        )
        return StoredPrediction(self.bucket, path)

    def _ensure_bucket(self) -> None:
        if self._client.bucket_exists(self.bucket):
            return
        try:
            self._client.make_bucket(self.bucket)
        except S3Error as error:
            # Another session may have created the bucket in the meantime.
            if error.code not in _ALREADY_EXISTS_CODES:
                raise
        log_event(logger, "bucket_ready", bucket=self.bucket)

    def _failure_message(self, error: Exception) -> str:
        invalid_name = isinstance(error, S3Error) and error.code == "InvalidBucketName"
        if isinstance(error, ValueError) or invalid_name:
            return (
                f"The result was not saved: '{self.bucket}' is not a valid bucket name. "
                "Check MINIO_BUCKET in your .env file. Use lowercase letters, numbers, and hyphens."
            )
        if isinstance(error, S3Error) and error.code in _AUTHENTICATION_CODES:
            return (
                "The result was not saved: MinIO refused the credentials. "
                "Check MINIO_ACCESS_KEY and MINIO_SECRET_KEY in your .env file."
            )
        if isinstance(error, S3Error):
            return f"The result was not saved: MinIO returned the error {error.code}."
        return (
            f"The result was not saved: MinIO could not be reached at {self._endpoint}. "
            "Start it with `docker compose up -d minio minio-console`, then try again."
        )
