"""Zero-shot inference using one loaded model at a time."""

import gc
import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.services.label_manager import validate_labels
from app.services.model_manager import ModelManager, ModelStatus
from app.utils.errors import ClassificationError, InvalidInputError, ModelNotReadyError
from app.utils.logging_config import log_event

logger = logging.getLogger(__name__)

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class RankedLabel:
    label: str
    score: float


@dataclass(frozen=True)
class ClassificationResult:
    request_id: str
    timestamp: str
    input_text: str
    predicted_class: str
    confidence: float
    candidate_labels: tuple[str, ...]
    ranked_predictions: tuple[RankedLabel, ...]
    model_name: str
    model_id: str
    model_revision: str | None
    latency_ms: float


def _create_pipeline(path: Path) -> Any:
    # Imported here so that pages open quickly and HF_HOME is already set.
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    return pipeline("zero-shot-classification", model=str(path), tokenizer=str(path), device=device)


def _ensure_pytorch(zero_shot_pipeline: Any) -> None:
    import torch

    if not isinstance(getattr(zero_shot_pipeline, "model", None), torch.nn.Module):
        raise ClassificationError("The model did not load on the required PyTorch backend.")


class Classifier:
    def __init__(
        self,
        models: ModelManager,
        max_input_chars: int,
        pipeline_factory: Callable[[Path], Any] = _create_pipeline,
    ) -> None:
        self._models = models
        self._max_input_chars = max_input_chars
        self._pipeline_factory = pipeline_factory
        self._pipeline: Any = None
        self._loaded_model_id: str | None = None
        self._lock = threading.Lock()

    def validate_text(self, text: object) -> str:
        cleaned = text.strip() if isinstance(text, str) else ""
        if not cleaned:
            raise InvalidInputError("Enter some text to classify.")
        if len(cleaned) > self._max_input_chars:
            raise InvalidInputError(
                f"The text has {len(cleaned):,} characters, "
                f"but the limit is {self._max_input_chars:,}. Shorten it and try again."
            )
        return cleaned

    def classify(
        self, text: object, model_id: str, labels: Iterable[object]
    ) -> ClassificationResult:
        cleaned_text = self.validate_text(text)
        candidate_labels = validate_labels(labels)
        model = self._models.status(model_id)
        if model.status is not ModelStatus.READY:
            raise ModelNotReadyError(
                f"{model.config.name} is not ready. Download it on the Admin page, "
                "then choose it again."
            )

        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        try:
            with self._lock:
                zero_shot_pipeline = self._load(model_id, model.path)
                output = zero_shot_pipeline(
                    cleaned_text, candidate_labels=list(candidate_labels), multi_label=False
                )
        except (RuntimeError, ValueError, OSError, MemoryError) as error:
            log_event(
                logger,
                "inference_failed",
                level=logging.ERROR,
                request_id=request_id,
                model_id=model_id,
                error_type=type(error).__name__,
            )
            raise ClassificationError(
                f"{model.config.name} could not classify this text. Check that the computer has "
                "enough memory. If this continues, delete the model and download it again."
            ) from None

        latency_ms = (time.perf_counter() - started) * 1000
        ranked = sorted(
            (
                RankedLabel(label, float(score))
                for label, score in zip(output["labels"], output["scores"], strict=True)
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        log_event(
            logger,
            "inference_completed",
            request_id=request_id,
            model_id=model_id,
            model_revision=model.revision,
            duration_ms=round(latency_ms, 1),
            label_count=len(candidate_labels),
        )
        return ClassificationResult(
            request_id=request_id,
            timestamp=datetime.now(UTC).strftime(TIMESTAMP_FORMAT),
            input_text=cleaned_text,
            predicted_class=ranked[0].label,
            confidence=ranked[0].score,
            candidate_labels=candidate_labels,
            ranked_predictions=tuple(ranked),
            model_name=model.config.name,
            model_id=model_id,
            model_revision=model.revision,
            latency_ms=latency_ms,
        )

    def _load(self, model_id: str, path: Path) -> Any:
        if self._loaded_model_id == model_id:
            return self._pipeline
        self._release()
        zero_shot_pipeline = self._pipeline_factory(path)
        _ensure_pytorch(zero_shot_pipeline)
        self._pipeline = zero_shot_pipeline
        self._loaded_model_id = model_id
        return zero_shot_pipeline

    def _release(self) -> None:
        self._pipeline = None
        self._loaded_model_id = None
        gc.collect()
