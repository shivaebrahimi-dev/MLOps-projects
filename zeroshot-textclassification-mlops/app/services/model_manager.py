"""Model configuration and local model status."""

import ctypes
import errno
import json
import logging
import os
import re
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import yaml

from app.utils.errors import (
    ConfigurationError,
    ModelDownloadError,
    ModelInUseError,
    ModelNotReadyError,
    ModelOperationError,
)
from app.utils.logging_config import log_event

logger = logging.getLogger(__name__)

ZERO_SHOT_TASK = "zero-shot-classification"
COMPLETION_MARKER = ".download-complete.json"
REQUIRED_MODEL_FILES = ("config.json", "model.safetensors")
TOKENIZER_FILES = ("tokenizer.json", "vocab.json", "spm.model")
# Only what PyTorch inference needs. Repositories also hold ONNX, Flax, and pickle-format files.
DOWNLOAD_PATTERNS = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spm.model",
    "vocab.json",
    "merges.txt",
)
TEMP_DIR_NAME = ".tmp"
# HF_HOME can also hold a saved login token, so only these download caches are cleared.
CACHE_FOLDERS = ("hub", "xet")

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


@dataclass(frozen=True)
class DownloadOutcome:
    model: LocalModel
    already_available: bool


@dataclass(frozen=True)
class CacheCleanup:
    removed_folders: tuple[str, ...]
    freed_bytes: int


class IncompleteDownloadError(Exception):
    """The download finished without all the files a model needs."""


Downloader = Callable[[str, str, Path, Sequence[str]], None]


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


def _download_snapshot(
    model_id: str, revision: str, local_dir: Path, patterns: Sequence[str]
) -> None:
    # Imported here so that pages open quickly and HF_HOME is already set.
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model_id, revision=revision, local_dir=local_dir, allow_patterns=list(patterns)
    )


class ModelManager:
    def __init__(
        self,
        configs: list[ModelConfig],
        model_dir: Path,
        hf_home: Path,
        downloader: Downloader = _download_snapshot,
    ) -> None:
        self._configs = {config.id: config for config in configs}
        self.model_dir = Path(model_dir).resolve()
        self.hf_home = Path(hf_home)
        self._downloader = downloader

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

    def download(self, model_id: str) -> DownloadOutcome:
        """Download a model unless it is ready. A partial download never becomes ready."""
        config = self.get_config(model_id)
        current = self.status(model_id)
        if current.status is ModelStatus.READY:
            log_event(
                logger, "model_download_skipped", model_id=model_id, revision=current.revision
            )
            return DownloadOutcome(current, already_available=True)
        self._acquire_lock(config)

        temp_path = self._temp_path(model_id, ".partial")
        started = time.perf_counter()
        log_event(logger, "model_download_started", model_id=model_id, revision=config.revision)
        try:
            self._install(config, temp_path)
        except Exception as error:
            message = self._download_failure_message(config, error)
            self._record_failure(model_id, message)
            log_event(
                logger,
                "model_download_failed",
                level=logging.ERROR,
                model_id=model_id,
                revision=config.revision,
                duration_ms=round((time.perf_counter() - started) * 1000),
                error_type=type(error).__name__,
            )
            raise ModelDownloadError(message) from None
        finally:
            # Also runs if Streamlit interrupts the script because the user left the page.
            shutil.rmtree(temp_path, ignore_errors=True)
            self._temp_path(model_id, ".lock").unlink(missing_ok=True)

        log_event(
            logger,
            "model_download_completed",
            model_id=model_id,
            revision=config.revision,
            duration_ms=round((time.perf_counter() - started) * 1000),
        )
        return DownloadOutcome(self.status(model_id), already_available=False)

    def delete(self, model_id: str) -> None:
        """Remove a local model. Unload it from the classifier first."""
        config = self.get_config(model_id)
        path = self.model_path(model_id)
        if self._download_in_progress(model_id):
            raise ModelInUseError(f"{config.name} is being downloaded, so it cannot be deleted.")
        failure_record = self._temp_path(model_id, ".failed.json")
        if not path.exists() and not failure_record.exists():
            raise ModelOperationError(f"{config.name} is not downloaded, so nothing was deleted.")
        try:
            if path.exists():
                shutil.rmtree(path)
            failure_record.unlink(missing_ok=True)
            self._remove_folder_if_empty(path.parent)
        except OSError as error:
            log_event(
                logger,
                "model_delete_failed",
                level=logging.ERROR,
                model_id=model_id,
                error_type=type(error).__name__,
            )
            raise ModelOperationError(
                f"{config.name} could not be fully deleted. Close anything using its files, "
                "then try again."
            ) from None
        log_event(logger, "model_deleted", model_id=model_id)

    def clear_cache(self) -> CacheCleanup:
        """Remove Hugging Face download caches, keeping saved tokens and downloaded models."""
        removed: list[str] = []
        freed = 0
        try:
            for name in CACHE_FOLDERS:
                folder = self.hf_home / name
                if folder.is_dir():
                    size = self._folder_size(folder)
                    shutil.rmtree(folder)
                    removed.append(name)
                    freed += size
        except OSError as error:
            log_event(
                logger,
                "cache_cleanup_failed",
                level=logging.ERROR,
                removed_folders=removed,
                error_type=type(error).__name__,
            )
            raise ModelOperationError(
                "The cache could not be fully cleared. Close anything using its files, "
                "then try again."
            ) from None
        log_event(logger, "cache_cleared", removed_folders=removed, freed_bytes=freed)
        return CacheCleanup(tuple(removed), freed)

    def _remove_folder_if_empty(self, folder: Path) -> None:
        inside_model_dir = folder != self.model_dir and folder.is_relative_to(self.model_dir)
        if inside_model_dir and folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()

    @staticmethod
    def _folder_size(folder: Path) -> int:
        return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())

    def _acquire_lock(self, config: ModelConfig) -> None:
        lock_path = self._temp_path(config.id, ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._download_in_progress(config.id):
            lock_path.unlink(missing_ok=True)
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise ModelDownloadError(
                f"{config.name} is already being downloaded. Wait for it to finish, "
                "then refresh the list."
            ) from None
        with os.fdopen(descriptor, "w", encoding="utf-8") as lock_file:
            json.dump({"pid": os.getpid()}, lock_file)

    def _install(self, config: ModelConfig, temp_path: Path) -> None:
        self._temp_path(config.id, ".failed.json").unlink(missing_ok=True)
        final_path = self.model_path(config.id)
        shutil.rmtree(temp_path, ignore_errors=True)
        self._downloader(config.id, config.revision, temp_path, DOWNLOAD_PATTERNS)

        missing = [name for name in REQUIRED_MODEL_FILES if not (temp_path / name).is_file()]
        if not any((temp_path / name).is_file() for name in TOKENIZER_FILES):
            missing.append("tokenizer")
        if missing:
            raise IncompleteDownloadError(", ".join(missing))

        # Holds Hugging Face download metadata that is not part of the model.
        shutil.rmtree(temp_path / ".cache", ignore_errors=True)
        (temp_path / COMPLETION_MARKER).write_text(
            json.dumps(
                {
                    "model_id": config.id,
                    "revision": config.revision,
                    "downloaded_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            ),
            encoding="utf-8",
        )
        # A leftover incomplete copy is never ready, so it is safe to replace.
        shutil.rmtree(final_path, ignore_errors=True)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.rename(final_path)

    def _record_failure(self, model_id: str, message: str) -> None:
        self._temp_path(model_id, ".failed.json").write_text(
            json.dumps({"error": message}), encoding="utf-8"
        )

    @staticmethod
    def _download_failure_message(config: ModelConfig, error: Exception) -> str:
        """Build a message from the error's type, because its text may include secrets."""
        error_types = {cls.__name__ for cls in type(error).__mro__}
        if "IncompleteDownloadError" in error_types:
            reason = "The repository did not provide all the files this application needs."
        elif "GatedRepoError" in error_types:
            reason = "This model needs approved access. Set HF_TOKEN in your .env file."
        elif "RevisionNotFoundError" in error_types:
            reason = f"Revision {config.revision} was not found. Check configs/models.yaml."
        elif "RepositoryNotFoundError" in error_types:
            reason = "The model was not found. Check its id in configs/models.yaml."
        elif isinstance(error, OSError) and error.errno == errno.ENOSPC:
            reason = "The disk is full. Free some space and try again."
        else:
            reason = "Check your internet connection and available disk space."
        return f"{config.name} could not be downloaded. {reason} You can retry the download."

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
