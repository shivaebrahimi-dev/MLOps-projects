import pytest

from app.services.model_manager import ModelConfig, ModelManager, ModelStatus
from app.utils.errors import ModelDownloadError

pytestmark = pytest.mark.integration

# Tiny repository with no `model.safetensors`, so the real downloader runs without large files.
NO_SAFETENSORS = ModelConfig(
    "hf-internal-testing/tiny-random-bart",
    "Tiny BART",
    "zero-shot-classification",
    "f2efe525625e508121ef8e13b7c37e6324073378",
)
UNKNOWN_REPOSITORY = ModelConfig(
    "hf-internal-testing/this-repository-does-not-exist",
    "Missing",
    "zero-shot-classification",
    "main",
)


def test_real_download_without_required_files_is_not_ready(tmp_path):
    manager = ModelManager([NO_SAFETENSORS], tmp_path / "models", tmp_path / "cache")

    with pytest.raises(ModelDownloadError, match="did not provide all the files"):
        manager.download(NO_SAFETENSORS.id)

    assert manager.status(NO_SAFETENSORS.id).status is ModelStatus.FAILED
    assert not manager.model_path(NO_SAFETENSORS.id).exists()
    temp_files = [path.name for path in (manager.model_dir / ".tmp").iterdir()]
    assert temp_files == ["hf-internal-testing--tiny-random-bart.failed.json"]


def test_real_download_of_unknown_repository_gives_a_clear_error(tmp_path):
    manager = ModelManager([UNKNOWN_REPOSITORY], tmp_path / "models", tmp_path / "cache")

    with pytest.raises(ModelDownloadError, match="could not be downloaded"):
        manager.download(UNKNOWN_REPOSITORY.id)

    assert manager.status(UNKNOWN_REPOSITORY.id).status is ModelStatus.FAILED
