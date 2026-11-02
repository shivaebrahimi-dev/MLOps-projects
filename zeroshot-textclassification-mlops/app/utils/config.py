"""Application settings loaded from the environment and an optional .env file."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from app.utils.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BUCKET = "zeroshot-learning-output"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Settings:
    minio_endpoint: str
    minio_access_key: str = field(repr=False)
    minio_secret_key: str = field(repr=False)
    minio_secure: bool
    minio_bucket: str
    model_dir: Path
    hf_home: Path
    hf_token: str | None = field(repr=False)
    models_config: Path
    labels_config: Path
    max_input_chars: int
    app_version: str
    log_level: str


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ConfigurationError(f"{name} is not set. Add it to your .env file.")
    return value


def _boolean(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name, "").strip().lower()
    if not value:
        return default
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ConfigurationError(f"{name} must be true or false.")


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    value = env.get(name, "").strip()
    if not value:
        return default
    try:
        number = int(value)
    except ValueError:
        raise ConfigurationError(f"{name} must be a whole number.") from None
    if number <= 0:
        raise ConfigurationError(f"{name} must be greater than zero.")
    return number


def _path(env: Mapping[str, str], name: str, default: str) -> Path:
    path = Path(env.get(name, "").strip() or default)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_settings(env: Mapping[str, str]) -> Settings:
    """Validate raw values and build settings. Error messages never include values."""
    return Settings(
        minio_endpoint=env.get("MINIO_ENDPOINT", "").strip() or "localhost:9000",
        minio_access_key=_required(env, "MINIO_ACCESS_KEY"),
        minio_secret_key=_required(env, "MINIO_SECRET_KEY"),
        minio_secure=_boolean(env, "MINIO_SECURE", default=False),
        minio_bucket=env.get("MINIO_BUCKET", "").strip() or DEFAULT_BUCKET,
        model_dir=_path(env, "MODEL_DIR", "models"),
        hf_home=_path(env, "HF_HOME", ".cache/huggingface"),
        hf_token=env.get("HF_TOKEN", "").strip() or None,
        models_config=_path(env, "MODELS_CONFIG", "configs/models.yaml"),
        labels_config=_path(env, "LABELS_CONFIG", "configs/labels.yaml"),
        max_input_chars=_positive_int(env, "MAX_INPUT_CHARS", default=10000),
        app_version=env.get("APP_VERSION", "").strip() or "0.1.0",
        log_level=env.get("LOG_LEVEL", "").strip().upper() or "INFO",
    )


def load_settings() -> Settings:
    """Load settings and set HF_HOME. Call this before importing Hugging Face libraries."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    settings = parse_settings(os.environ)
    os.environ["HF_HOME"] = str(settings.hf_home)
    return settings
