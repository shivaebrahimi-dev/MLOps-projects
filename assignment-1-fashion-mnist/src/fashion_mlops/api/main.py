"""FastAPI prediction service.

    GET  /health    liveness plus whether a model is currently available
    POST /predict   multipart upload of one image (form field ``file``)

Run locally with ``uvicorn fashion_mlops.api.main:app``.
"""

from __future__ import annotations

import io
import logging
import threading
from collections.abc import Callable
from functools import lru_cache

import torch
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from fashion_mlops import storage
from fashion_mlops.config import ServiceSettings
from fashion_mlops.model import CLASS_NAMES, LeNet5, deserialize_model
from fashion_mlops.preprocessing import preprocess_image

logger = logging.getLogger("fashion_mlops.api")

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


class ModelService:
    """Loads the selected model from MinIO and caches it between requests.

    Which model is "selected" is decided by ``storage.resolve_model_key``: the
    pinned ``MODEL_OBJECT_KEY`` if set, otherwise whatever ``model/latest.json``
    points at. The pointer is re-read on every call, so a newly trained model is
    picked up without restarting the service.
    """

    def __init__(
        self,
        settings: ServiceSettings,
        client_factory: Callable[[ServiceSettings], object] = storage.make_client,
    ) -> None:
        self.settings = settings
        self._client_factory = client_factory
        self._lock = threading.Lock()
        self._model: LeNet5 | None = None
        self._key: str | None = None

    def get(self) -> tuple[LeNet5, str]:
        """Return (model, object key), raising ``StorageError`` if none can be provided."""
        with self._lock:
            try:
                client = self._client_factory(self.settings)
                key = storage.resolve_model_key(client, self.settings)
                if key != self._key:
                    data = storage.download_model(client, self.settings, key)
                    try:
                        self._model = deserialize_model(data)
                    except Exception as exc:
                        raise storage.StorageError(
                            f"'{key}' could not be loaded as a model: {exc}"
                        ) from exc
                    self._key = key
                    logger.info("loaded model %s", key)
            except storage.StorageError:
                # A model that is already in memory keeps serving through a storage outage.
                if self._model is None:
                    raise
                logger.warning("storage unavailable; serving cached model %s", self._key)
            return self._model, self._key


@lru_cache
def get_model_service() -> ModelService:
    return ModelService(ServiceSettings())


class HealthResponse(BaseModel):
    status: str
    model_available: bool
    model_object_key: str | None = None
    detail: str | None = None


class PredictionResponse(BaseModel):
    predicted_class: str
    class_index: int
    confidence: float
    probabilities: dict[str, float]
    model_object_key: str


app = FastAPI(
    title="Fashion-MNIST classifier",
    description="Predicts the Fashion-MNIST class of a single uploaded image.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health(service: ModelService = Depends(get_model_service)) -> HealthResponse:
    """Always 200 while the service is up; ``model_available`` says whether it can predict."""
    try:
        _, key = service.get()
    except storage.StorageError as exc:
        return HealthResponse(status="ok", model_available=False, detail=str(exc))
    return HealthResponse(status="ok", model_available=True, model_object_key=key)


def read_image(upload: UploadFile) -> Image.Image:
    """Validate an upload and decode it, raising a 4xx ``HTTPException`` if it is unusable."""
    if not (upload.content_type or "").startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{upload.content_type}'. Upload an image (PNG, JPEG, ...).",
        )
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than the 5 MB limit.")
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(
            status_code=400, detail="The file could not be read as an image."
        ) from exc
    return image


@app.post("/predict", response_model=PredictionResponse)
def predict(
    file: UploadFile = File(..., description="One image; it is converted to 28x28 grayscale."),
    service: ModelService = Depends(get_model_service),
) -> PredictionResponse:
    image = read_image(file)
    try:
        model, key = service.get()
    except storage.StorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    with torch.no_grad():
        probabilities = torch.softmax(model(preprocess_image(image)), dim=1)[0]
    index = int(probabilities.argmax())
    return PredictionResponse(
        predicted_class=CLASS_NAMES[index],
        class_index=index,
        confidence=float(probabilities[index]),
        probabilities={name: float(p) for name, p in zip(CLASS_NAMES, probabilities)},
        model_object_key=key,
    )
