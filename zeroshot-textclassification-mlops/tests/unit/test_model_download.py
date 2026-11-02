import errno
import json
import os

import pytest

from app.services.model_manager import (
    COMPLETION_MARKER,
    DOWNLOAD_PATTERNS,
    ModelManager,
    ModelStatus,
    load_model_configs,
)
from app.utils.errors import ModelDownloadError
from tests.helpers import MODEL, make_ready_model


class FakeDownloader:
    """Writes model files, optionally stopping partway through with an error."""

    def __init__(self, failure=None, files=("config.json", "model.safetensors", "tokenizer.json")):
        self.failure = failure
        self.files = files
        self.calls = []

    def __call__(self, model_id, revision, local_dir, patterns):
        self.calls.append((model_id, revision, patterns))
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").write_text("{}")
        if self.failure:
            raise self.failure
        (local_dir / ".cache").mkdir()
        (local_dir / ".cache" / "metadata").write_text("x")
        for name in self.files:
            (local_dir / name).write_text("{}")


def manager_with(tmp_path, downloader):
    return ModelManager([MODEL], tmp_path / "models", tmp_path / "cache", downloader=downloader)


def temp_files(manager):
    temp_dir = manager.model_dir / ".tmp"
    return sorted(path.name for path in temp_dir.iterdir()) if temp_dir.exists() else []


def test_missing_model_is_downloaded_and_becomes_ready(tmp_path):
    downloader = FakeDownloader()
    manager = manager_with(tmp_path, downloader)

    outcome = manager.download(MODEL.id)

    assert outcome.already_available is False
    assert outcome.model.status is ModelStatus.READY
    assert outcome.model.revision == MODEL.revision
    assert outcome.model.downloaded_at
    assert outcome.model.path == manager.model_dir / "org" / "model"
    assert downloader.calls == [(MODEL.id, MODEL.revision, DOWNLOAD_PATTERNS)]
    assert temp_files(manager) == []


def test_only_needed_files_are_requested():
    assert "model.safetensors" in DOWNLOAD_PATTERNS
    for unwanted in ("pytorch_model.bin", "training_args.bin", "flax_model.msgpack", "onnx/*"):
        assert unwanted not in DOWNLOAD_PATTERNS


def test_download_metadata_is_not_kept_with_the_model(tmp_path):
    manager = manager_with(tmp_path, FakeDownloader())

    outcome = manager.download(MODEL.id)

    assert not (outcome.model.path / ".cache").exists()
    marker = json.loads((outcome.model.path / COMPLETION_MARKER).read_text())
    assert marker["model_id"] == MODEL.id
    assert marker["revision"] == MODEL.revision


def test_ready_model_is_not_downloaded_again(tmp_path):
    downloader = FakeDownloader()
    manager = manager_with(tmp_path, downloader)
    manager.download(MODEL.id)

    outcome = manager.download(MODEL.id)

    assert outcome.already_available is True
    assert len(downloader.calls) == 1


def test_failed_download_leaves_no_model_or_temporary_files(tmp_path):
    manager = manager_with(tmp_path, FakeDownloader(failure=ConnectionError("network down")))

    with pytest.raises(ModelDownloadError, match="could not be downloaded"):
        manager.download(MODEL.id)

    model = manager.status(MODEL.id)
    assert model.status is ModelStatus.FAILED
    assert "retry" in model.error
    assert not model.path.exists()
    assert temp_files(manager) == ["org--model.failed.json"]


def test_download_missing_required_files_is_never_ready(tmp_path):
    manager = manager_with(tmp_path, FakeDownloader(files=("config.json", "tokenizer.json")))

    with pytest.raises(ModelDownloadError, match="did not provide all the files"):
        manager.download(MODEL.id)

    assert manager.status(MODEL.id).status is ModelStatus.FAILED
    assert not manager.model_path(MODEL.id).exists()


def test_download_without_a_tokenizer_is_never_ready(tmp_path):
    manager = manager_with(tmp_path, FakeDownloader(files=("config.json", "model.safetensors")))

    with pytest.raises(ModelDownloadError, match="did not provide all the files"):
        manager.download(MODEL.id)


def test_retry_after_failure_succeeds_and_clears_the_failure(tmp_path):
    downloader = FakeDownloader(failure=ConnectionError("network down"))
    manager = manager_with(tmp_path, downloader)
    with pytest.raises(ModelDownloadError):
        manager.download(MODEL.id)

    downloader.failure = None
    outcome = manager.download(MODEL.id)

    assert outcome.model.status is ModelStatus.READY
    assert outcome.model.error is None
    assert temp_files(manager) == []


def test_incomplete_final_folder_is_replaced_by_a_new_download(tmp_path):
    manager = manager_with(tmp_path, FakeDownloader())
    leftover = make_ready_model(manager)
    (leftover / COMPLETION_MARKER).unlink()
    (leftover / "stale-file").write_text("old")
    assert manager.status(MODEL.id).status is ModelStatus.FAILED

    outcome = manager.download(MODEL.id)

    assert outcome.model.status is ModelStatus.READY
    assert not (outcome.model.path / "stale-file").exists()


def test_leftover_partial_folder_is_discarded(tmp_path):
    manager = manager_with(tmp_path, FakeDownloader())
    partial = manager.model_dir / ".tmp" / "org--model.partial"
    partial.mkdir(parents=True)
    (partial / "stale-file").write_text("old")

    outcome = manager.download(MODEL.id)

    assert not (outcome.model.path / "stale-file").exists()


def test_second_download_while_one_is_running_is_refused(tmp_path):
    downloader = FakeDownloader()
    manager = manager_with(tmp_path, downloader)
    lock = manager.model_dir / ".tmp" / "org--model.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({"pid": os.getpid()}))

    with pytest.raises(ModelDownloadError, match="already being downloaded"):
        manager.download(MODEL.id)

    assert downloader.calls == []
    assert lock.exists()


def test_status_is_downloading_during_the_download(tmp_path):
    seen = []

    def downloader(model_id, revision, local_dir, patterns):
        seen.append(manager.status(model_id).status)
        raise ConnectionError("stop here")

    manager = manager_with(tmp_path, downloader)
    with pytest.raises(ModelDownloadError):
        manager.download(MODEL.id)

    assert seen == [ModelStatus.DOWNLOADING]


def test_stale_lock_is_removed(tmp_path):
    manager = manager_with(tmp_path, FakeDownloader())
    lock = manager.model_dir / ".tmp" / "org--model.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({"pid": 2_000_000_000}))

    outcome = manager.download(MODEL.id)

    assert outcome.model.status is ModelStatus.READY
    assert not lock.exists()


def test_interrupted_download_cleans_up(tmp_path):
    class ScriptStopped(BaseException):
        """Like the exception Streamlit raises when a user leaves a running page."""

    manager = manager_with(tmp_path, FakeDownloader(failure=ScriptStopped()))

    with pytest.raises(ScriptStopped):
        manager.download(MODEL.id)

    assert temp_files(manager) == []
    assert manager.status(MODEL.id).status is ModelStatus.NOT_DOWNLOADED


@pytest.mark.parametrize(
    ("error_name", "message"),
    [
        ("GatedRepoError", "HF_TOKEN"),
        ("RepositoryNotFoundError", "not found"),
        ("RevisionNotFoundError", "abc123"),
    ],
)
def test_hugging_face_errors_get_specific_messages(tmp_path, error_name, message):
    failure = type(error_name, (Exception,), {})("details hf_secretvalue123")
    manager = manager_with(tmp_path, FakeDownloader(failure=failure))

    with pytest.raises(ModelDownloadError, match=message) as error:
        manager.download(MODEL.id)

    assert "hf_secretvalue123" not in str(error.value)
    assert "hf_secretvalue123" not in manager.status(MODEL.id).error


def test_full_disk_gets_a_specific_message(tmp_path):
    failure = OSError(errno.ENOSPC, "No space left")
    manager = manager_with(tmp_path, FakeDownloader(failure=failure))

    with pytest.raises(ModelDownloadError, match="disk is full"):
        manager.download(MODEL.id)


def test_newly_configured_model_appears_without_code_changes(tmp_path):
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "models:\n"
        "  - {id: new-org/new-model, name: New, task: zero-shot-classification, revision: abc}\n"
    )
    manager = ModelManager(
        load_model_configs(config_path),
        tmp_path / "models",
        tmp_path / "cache",
        downloader=FakeDownloader(),
    )

    assert [model.config.id for model in manager.statuses()] == ["new-org/new-model"]
    assert manager.download("new-org/new-model").model.status is ModelStatus.READY
