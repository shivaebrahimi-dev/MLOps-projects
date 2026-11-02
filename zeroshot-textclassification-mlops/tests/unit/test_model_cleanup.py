import json
import os
import shutil

import pytest

from app.services.classifier import Classifier
from app.services.model_manager import ModelManager, ModelStatus
from app.utils.errors import ModelInUseError, ModelOperationError
from tests.helpers import MODEL, FakePipeline, make_ready_model

OTHER = type(MODEL)("org/other", "Other", MODEL.task, "def456")


@pytest.fixture
def manager(tmp_path):
    return ModelManager([MODEL, OTHER], tmp_path / "models", tmp_path / "cache")


def test_deleted_model_is_no_longer_available(manager):
    path = make_ready_model(manager)

    manager.delete(MODEL.id)

    assert not path.exists()
    assert manager.status(MODEL.id).status is ModelStatus.NOT_DOWNLOADED
    assert manager.ready_models() == []


def test_deleting_one_model_keeps_the_others(manager):
    make_ready_model(manager)
    other_path = make_ready_model(manager, OTHER.id, OTHER.revision)

    manager.delete(MODEL.id)

    assert other_path.exists()
    assert manager.status(OTHER.id).status is ModelStatus.READY


def test_empty_organization_folder_is_removed(manager):
    path = make_ready_model(manager)

    manager.delete(MODEL.id)

    assert not path.parent.exists()
    assert manager.model_dir.exists()


def test_deleting_a_missing_model_is_reported(manager):
    with pytest.raises(ModelOperationError, match="nothing was deleted"):
        manager.delete(MODEL.id)


def test_failed_model_can_be_cleared(manager):
    temp_dir = manager.model_dir / ".tmp"
    temp_dir.mkdir(parents=True)
    (temp_dir / "org--model.failed.json").write_text(json.dumps({"error": "Failed."}))

    manager.delete(MODEL.id)

    assert manager.status(MODEL.id).status is ModelStatus.NOT_DOWNLOADED


def test_model_being_downloaded_cannot_be_deleted(manager):
    path = make_ready_model(manager)
    temp_dir = manager.model_dir / ".tmp"
    temp_dir.mkdir(parents=True)
    (temp_dir / "org--model.lock").write_text(json.dumps({"pid": os.getpid()}))

    with pytest.raises(ModelInUseError, match="being downloaded"):
        manager.delete(MODEL.id)

    assert path.exists()


def test_failed_deletion_gives_a_clear_error(manager, monkeypatch):
    make_ready_model(manager)

    def fail(path):
        raise PermissionError("locked")

    monkeypatch.setattr(shutil, "rmtree", fail)

    with pytest.raises(ModelOperationError, match="could not be fully deleted"):
        manager.delete(MODEL.id)


@pytest.mark.parametrize("model_id", ["../escape", "org/../../escape"])
def test_unconfigured_paths_cannot_be_deleted(manager, tmp_path, model_id):
    outside = tmp_path / "escape"
    outside.mkdir()

    with pytest.raises(Exception, match="not configured"):
        manager.delete(model_id)

    assert outside.exists()


def test_cleanup_handles_a_missing_cache(manager):
    cleanup = manager.clear_cache()

    assert cleanup.removed_folders == ()
    assert cleanup.freed_bytes == 0


def test_cleanup_removes_caches_and_reports_size(manager):
    for name in ("hub", "xet"):
        folder = manager.hf_home / name / "nested"
        folder.mkdir(parents=True)
        (folder / "blob").write_bytes(b"x" * 100)

    cleanup = manager.clear_cache()

    assert cleanup.removed_folders == ("hub", "xet")
    assert cleanup.freed_bytes == 200
    assert not (manager.hf_home / "hub").exists()
    assert not (manager.hf_home / "xet").exists()


def test_cleanup_keeps_saved_tokens(manager):
    (manager.hf_home / "hub").mkdir(parents=True)
    token = manager.hf_home / "token"
    token.write_text("saved-login")

    manager.clear_cache()

    assert token.read_text() == "saved-login"


def test_ready_models_survive_cache_cleanup(manager):
    make_ready_model(manager)
    (manager.hf_home / "hub").mkdir(parents=True)

    manager.clear_cache()

    assert manager.status(MODEL.id).status is ModelStatus.READY


def test_failed_cleanup_gives_a_clear_error(manager, monkeypatch):
    (manager.hf_home / "hub").mkdir(parents=True)

    def fail(path):
        raise PermissionError("locked")

    monkeypatch.setattr(shutil, "rmtree", fail)

    with pytest.raises(ModelOperationError, match="could not be fully cleared"):
        manager.clear_cache()


def test_unloading_releases_only_the_named_model(manager):
    make_ready_model(manager)
    pipeline = FakePipeline({"a": 0.9, "b": 0.1})
    classifier = Classifier(manager, 100, pipeline_factory=lambda path: pipeline)
    classifier.classify("Text.", MODEL.id, ["a", "b"])

    classifier.unload(OTHER.id)
    assert classifier._loaded_model_id == MODEL.id

    classifier.unload(MODEL.id)
    assert classifier._loaded_model_id is None
    assert classifier._pipeline is None
