import json
import os

import pytest

from app.services.model_manager import (
    COMPLETION_MARKER,
    ModelConfig,
    ModelManager,
    ModelStatus,
    is_valid_model_id,
    load_model_configs,
)
from app.utils.config import PROJECT_ROOT
from app.utils.errors import ConfigurationError, ModelNotReadyError

MODEL = ModelConfig("org/model", "Test Model", "zero-shot-classification", "abc123")
VALID_ENTRY = "- {id: org/model, name: Test, task: zero-shot-classification, revision: abc}"


@pytest.fixture
def manager(tmp_path):
    return ModelManager([MODEL], tmp_path / "models", tmp_path / "cache")


def make_ready(manager, model_id="org/model", marker_id=None):
    path = manager.model_path(model_id)
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_text("weights")
    (path / COMPLETION_MARKER).write_text(
        json.dumps(
            {
                "model_id": marker_id or model_id,
                "revision": "abc123",
                "downloaded_at": "2026-11-01T00:00:00Z",
            }
        )
    )
    return path


def write_config(tmp_path, text):
    path = tmp_path / "models.yaml"
    path.write_text(text)
    return path


def test_default_configuration_has_both_required_models():
    configs = load_model_configs(PROJECT_ROOT / "configs" / "models.yaml")

    ids = {config.id for config in configs}
    assert {"MoritzLaurer/deberta-v3-base-zeroshot-v2.0", "facebook/bart-large-mnli"} <= ids
    assert all(config.revision for config in configs)


def test_new_model_only_needs_configuration(tmp_path):
    path = write_config(tmp_path, f"models:\n  {VALID_ENTRY}\n")

    assert [config.id for config in load_model_configs(path)] == ["org/model"]


@pytest.mark.parametrize(
    "text",
    [
        "models: [",
        "models: []",
        "other: 1",
        "models:\n  - {id: org/model}",
        "models:\n  - {id: bad, name: Test, task: zero-shot-classification, revision: abc}",
        "models:\n  - {id: org/model, name: Test, task: translation, revision: abc}",
        f"models:\n  {VALID_ENTRY}\n  {VALID_ENTRY}",
    ],
)
def test_invalid_configuration_is_rejected(tmp_path, text):
    with pytest.raises(ConfigurationError, match="models.yaml"):
        load_model_configs(write_config(tmp_path, text))


def test_missing_configuration_file_is_reported(tmp_path):
    with pytest.raises(ConfigurationError, match="not found"):
        load_model_configs(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    "model_id",
    ["../escape", "org/..", "org/../../escape", "/absolute/path", "org\\name", "name", "a/b/c", ""],
)
def test_unsafe_model_ids_are_rejected(manager, model_id):
    assert not is_valid_model_id(model_id)
    with pytest.raises(ConfigurationError):
        manager.model_path(model_id)


def test_model_path_stays_inside_model_directory(manager):
    path = manager.model_path("org/model")

    assert path == manager.model_dir / "org" / "model"
    assert path.is_relative_to(manager.model_dir)


def test_missing_model_is_not_downloaded(manager):
    model = manager.status("org/model")

    assert model.status is ModelStatus.NOT_DOWNLOADED
    assert manager.ready_models() == []


def test_unknown_model_is_reported(manager):
    with pytest.raises(ModelNotReadyError, match="not configured"):
        manager.status("org/unknown")


def test_complete_model_is_ready_with_marker_details(manager):
    make_ready(manager)

    model = manager.status("org/model")

    assert model.status is ModelStatus.READY
    assert model.revision == "abc123"
    assert model.downloaded_at == "2026-11-01T00:00:00Z"
    assert [ready.config.id for ready in manager.ready_models()] == ["org/model"]


def test_folder_without_marker_is_never_ready(manager):
    path = make_ready(manager)
    (path / COMPLETION_MARKER).unlink()

    assert manager.status("org/model").status is ModelStatus.FAILED


def test_corrupt_marker_is_never_ready(manager):
    path = make_ready(manager)
    (path / COMPLETION_MARKER).write_text("{not json")

    assert manager.status("org/model").status is ModelStatus.FAILED


def test_marker_for_another_model_is_never_ready(manager):
    make_ready(manager, marker_id="org/other")

    assert manager.status("org/model").status is ModelStatus.FAILED


def test_missing_weights_are_never_ready(manager):
    path = make_ready(manager)
    (path / "model.safetensors").unlink()

    assert manager.status("org/model").status is ModelStatus.FAILED


def write_temp_file(manager, suffix, content):
    temp_dir = manager.model_dir / ".tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / f"org--model{suffix}").write_text(json.dumps(content))


def test_lock_from_running_process_means_downloading(manager):
    write_temp_file(manager, ".lock", {"pid": os.getpid()})

    assert manager.status("org/model").status is ModelStatus.DOWNLOADING


def test_stale_lock_is_ignored(manager):
    # Far above any process ID the operating system assigns in practice.
    write_temp_file(manager, ".lock", {"pid": 2_000_000_000})

    assert manager.status("org/model").status is ModelStatus.NOT_DOWNLOADED


def test_recorded_failure_is_reported(manager):
    write_temp_file(manager, ".failed.json", {"error": "Network failure."})

    model = manager.status("org/model")

    assert model.status is ModelStatus.FAILED
    assert model.error == "Network failure."
