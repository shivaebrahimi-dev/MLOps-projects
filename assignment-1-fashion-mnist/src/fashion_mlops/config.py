"""Configuration for training runs and for the services around them.

Training settings (``TrainConfig``) are resolved from, highest priority first:
command-line overrides, ``TRAIN_*`` environment variables, a TOML file, defaults.

Service settings (``ServiceSettings``) hold endpoints and credentials and come
only from environment variables or a local ``.env`` file, never from source.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PREFIX = "TRAIN_"


class TrainConfig(BaseModel):
    """Everything needed to reproduce a training run."""

    # extra="forbid" turns a typo in the TOML file into an error instead of a silently ignored key.
    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(5, ge=1)
    batch_size: int = Field(64, ge=1)
    learning_rate: float = Field(1e-3, gt=0)
    optimizer: Literal["adam", "sgd"] = "adam"
    loss: Literal["cross_entropy", "nll"] = "cross_entropy"
    seed: int = 42
    val_split: float = Field(0.1, gt=0, lt=1)

    momentum: float = Field(0.9, ge=0, lt=1)
    weight_decay: float = Field(0.0, ge=0)
    early_stopping_patience: int = Field(0, ge=0)
    num_workers: int = Field(0, ge=0)
    data_dir: str = "data"
    experiment_name: str = "fashion-mnist-lenet5"


def load_train_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> TrainConfig:
    """Build a validated ``TrainConfig`` from a TOML file, the environment and overrides."""
    values: dict[str, Any] = {}

    if path is not None:
        with open(path, "rb") as f:
            values.update(tomllib.load(f).get("training", {}))

    env = os.environ if env is None else env
    for name in TrainConfig.model_fields:
        raw = env.get(ENV_PREFIX + name.upper())
        if raw not in (None, ""):
            values[name] = raw

    for name, value in (overrides or {}).items():
        if value is not None:
            values[name] = value

    return TrainConfig(**values)


class ServiceSettings(BaseSettings):
    """Endpoints and credentials shared by the trainer, the API and the UI."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_secure: bool = False
    # "minst" is the spelling required by the assignment.
    minio_bucket: str = "fashion-minst-bucket"
    model_prefix: str = "model"

    mlflow_tracking_uri: str = "http://localhost:5000"
    webhook_url: str = ""
    model_object_key: str = ""
    api_url: str = "http://localhost:8000"
