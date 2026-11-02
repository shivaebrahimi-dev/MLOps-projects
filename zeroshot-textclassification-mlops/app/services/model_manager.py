"""Model configuration and local model status."""

import ctypes
import json
import os
import re
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from app.utils.errors import ConfigurationError, ModelNotReadyError

ZERO_SHOT_TASK = "zero-shot-classification"
COMPLETION_MARKER = ".download-complete.json"
REQUIRED_MODEL_FILES = ("config.json", "model.safetensors")
TEMP_DIR_NAME = ".tmp"

_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_WINDOWS_QUERY_LIMITED_INFORMATION = 0x1000
_WINDOWS_STILL_ACTIVE = 259


class ModelStatus(StrEnum):
    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class ModelConfig:
    id: str
    name: str
    task: str
    revision: str


@dataclass(frozen=True)
class LocalModel:
    config: ModelConfig
    path: Path
    status: ModelStatus
    revision: str | None = None
    downloaded_at: str | None = None
    error: str | None = None


def is_valid_model_id(model_id: object) -> bool:
    """Accept only `org/name` identifiers that cannot leave the model directory."""
    return isinstance(model_id, str) and bool(_MODEL_ID.match(model_id)) and ".." not in model_id


def load_model_configs(path: Path) -> list[ModelConfig]:
    """Read and validate the model configuration file."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError:
        raise ConfigurationError(f"Model configuration file {path.name} was not found.") from None
    except yaml.YAMLError:
        raise ConfigurationError(
            f"Model configuration file {path.name} is not valid YAML."
        ) from None

    entries = data.get("models") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ConfigurationError(f"{path.name} must contain a non-empty `models` list.")

    configs: list[ModelConfig] = []
    seen: set[str] = set()
    for entry in entries:
        configs.append(_parse_model_entry(entry, path.name))
        if configs[-1].id in seen:
            raise ConfigurationError(f"{path.name} lists model {configs[-1].id} more than once.")
        seen.add(configs[-1].id)
    return configs


def _parse_model_entry(entry: object, file_name: str) -> ModelConfig:
    if not isinstance(entry, dict):
        raise ConfigurationError(f"{file_name} contains a model entry that is not a mapping.")
    model_id = entry.get("id")
    if not is_valid_model_id(model_id):
        raise ConfigurationError(
            f"{file_name} contains an invalid model id. Use the Hugging Face format org/name."
        )
    for field_name in ("name", "task", "revision"):
        value = entry.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(f"Model {model_id} in {file_name} needs a `{field_name}`.")
    if entry["task"] != ZERO_SHOT_TASK:
        raise ConfigurationError(f"Model {model_id} in {file_name} must use task {ZERO_SHOT_TASK}.")
    return ModelConfig(
        id=model_id, name=entry["name"], task=entry["task"], revision=entry["revision"]
    )


def _process_is_running(pid: int) -> bool:
    if sys.platform == "win32":
        # os.kill(pid, 0) would send Ctrl+C on Windows, so query the process instead.
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(_WINDOWS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            found = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            return bool(found) and exit_code.value == _WINDOWS_STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class ModelManager:
    def __init__(self, configs: list[ModelConfig], model_dir: Path, hf_home: Path) -> None:
        self._configs = {config.id: config for config in configs}
        self.model_dir = Path(model_dir).resolve()
        self.hf_home = Path(hf_home)

    def configs(self) -> list[ModelConfig]:
        return list(self._configs.values())

    def get_config(self, model_id: str) -> ModelConfig:
        try:
            return self._configs[model_id]
        except KeyError:
            raise ModelNotReadyError(f"Model {model_id} is not configured.") from None

    def model_path(self, model_id: str) -> Path:
        """Return the model's folder, refusing any path outside the model directory."""
        if not is_valid_model_id(model_id):
            raise ConfigurationError(f"Model id {model_id!r} is not valid.")
        path = (self.model_dir / model_id).resolve()
        if not path.is_relative_to(self.model_dir) or path == self.model_dir:
            raise ConfigurationError(f"Model id {model_id!r} is not valid.")
        return path

    def status(self, model_id: str) -> LocalModel:
        config = self.get_config(model_id)
        path = self.model_path(model_id)
        if self._download_in_progress(model_id):
            return LocalModel(config, path, ModelStatus.DOWNLOADING)
        if path.exists():
            return self._status_of_existing_folder(config, path)
        error = self._recorded_failure(model_id)
        if error is not None:
            return LocalModel(config, path, ModelStatus.FAILED, error=error)
        return LocalModel(config, path, ModelStatus.NOT_DOWNLOADED)

    def statuses(self) -> list[LocalModel]:
        return [self.status(model_id) for model_id in self._configs]

    def ready_models(self) -> list[LocalModel]:
        return [model for model in self.statuses() if model.status is ModelStatus.READY]

    def _status_of_existing_folder(self, config: ModelConfig, path: Path) -> LocalModel:
        incomplete = LocalModel(
            config,
            path,
            ModelStatus.FAILED,
            error="The local copy is incomplete. Delete it and download the model again.",
        )
        marker = self._read_json(path / COMPLETION_MARKER)
        if marker is None or marker.get("model_id") != config.id:
            return incomplete
        if not all((path / name).is_file() for name in REQUIRED_MODEL_FILES):
            return incomplete
        return LocalModel(
            config,
            path,
            ModelStatus.READY,
            revision=marker.get("revision"),
            downloaded_at=marker.get("downloaded_at"),
        )

    def _temp_path(self, model_id: str, suffix: str) -> Path:
        return self.model_dir / TEMP_DIR_NAME / f"{model_id.replace('/', '--')}{suffix}"

    def _download_in_progress(self, model_id: str) -> bool:
        lock = self._read_json(self._temp_path(model_id, ".lock"))
        if lock is None:
            return False
        pid = lock.get("pid")
        return isinstance(pid, int) and _process_is_running(pid)

    def _recorded_failure(self, model_id: str) -> str | None:
        failure = self._read_json(self._temp_path(model_id, ".failed.json"))
        return None if failure is None else str(failure.get("error", "The download failed."))

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None
