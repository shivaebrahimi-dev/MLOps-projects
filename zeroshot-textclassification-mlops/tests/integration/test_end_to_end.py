import json
import shutil

import pytest
from transformers import AutoConfig, AutoTokenizer, BartForSequenceClassification

from app.services.classifier import Classifier
from app.services.minio_storage import PredictionStorage, create_client
from app.services.model_manager import COMPLETION_MARKER, ModelConfig, ModelManager, ModelStatus

pytestmark = pytest.mark.integration

TINY_MODEL_ID = "hf-internal-testing/tiny-random-bart"
TINY_REVISION = "f2efe525625e508121ef8e13b7c37e6324073378"
LABELS = ["sports", "finance", "politics"]


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """Save a tiny, untrained model in the same layout as a downloaded one."""
    id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
    config = AutoConfig.from_pretrained(
        TINY_MODEL_ID,
        revision=TINY_REVISION,
        id2label=id2label,
        label2id={label: index for index, label in id2label.items()},
    )
    folder = tmp_path_factory.mktemp("models")
    path = folder / TINY_MODEL_ID
    BartForSequenceClassification(config).save_pretrained(path)
    AutoTokenizer.from_pretrained(TINY_MODEL_ID, revision=TINY_REVISION).save_pretrained(path)
    (path / COMPLETION_MARKER).write_text(
        json.dumps(
            {
                "model_id": TINY_MODEL_ID,
                "revision": TINY_REVISION,
                "downloaded_at": "2026-11-01T00:00:00Z",
            }
        )
    )
    return folder


def test_model_in_memory_can_be_unloaded_and_deleted(tiny_model_dir, tmp_path):
    # Windows refuses to delete files a loaded model still holds open.
    folder = tmp_path / "models"
    shutil.copytree(tiny_model_dir, folder)
    config = ModelConfig(TINY_MODEL_ID, "Tiny BART", "zero-shot-classification", TINY_REVISION)
    models = ModelManager([config], folder, tmp_path / "cache")
    classifier = Classifier(models, max_input_chars=1000)
    classifier.classify("The team won the championship.", TINY_MODEL_ID, LABELS)

    classifier.unload(TINY_MODEL_ID)
    models.delete(TINY_MODEL_ID)

    assert models.status(TINY_MODEL_ID).status is ModelStatus.NOT_DOWNLOADED
    assert not models.model_path(TINY_MODEL_ID).exists()


def test_text_is_classified_with_pytorch_and_stored_in_minio(
    minio_settings, tiny_model_dir, tmp_path
):
    config = ModelConfig(TINY_MODEL_ID, "Tiny BART", "zero-shot-classification", TINY_REVISION)
    models = ModelManager([config], tiny_model_dir, tmp_path / "cache")
    classifier = Classifier(models, max_input_chars=1000)
    storage = PredictionStorage.from_settings(minio_settings)

    result = classifier.classify("The team won the championship.", TINY_MODEL_ID, LABELS)
    stored = storage.save(result, session_id="end-to-end")

    record = json.loads(
        create_client(minio_settings).get_object(stored.bucket, stored.object_path).read()
    )
    assert record["request_id"] == result.request_id
    assert record["input_text"] == "The team won the championship."
    assert record["model"] == TINY_MODEL_ID
    assert record["model_revision"] == TINY_REVISION
    assert record["session_id"] == "end-to-end"
    assert sorted(item["label"] for item in record["ranked_predictions"]) == sorted(LABELS)
    assert record["predicted_class"] == record["ranked_predictions"][0]["label"]
    assert record["confidence"] == max(item["score"] for item in record["ranked_predictions"])
