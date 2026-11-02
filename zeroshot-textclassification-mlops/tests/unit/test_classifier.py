import uuid
from datetime import datetime

import pytest

from app.services.classifier import TIMESTAMP_FORMAT, Classifier
from app.services.model_manager import ModelManager
from app.utils.errors import (
    ClassificationError,
    InvalidInputError,
    LabelValidationError,
    ModelNotReadyError,
)
from tests.helpers import MODEL, FakePipeline, make_ready_model

SCORES = {"sports": 0.9, "finance": 0.07, "politics": 0.03}
LABELS = ["finance", "sports", "politics"]
OTHER = type(MODEL)("org/other", "Other Model", MODEL.task, "def456")


@pytest.fixture
def manager(tmp_path):
    return ModelManager([MODEL, OTHER], tmp_path / "models", tmp_path / "cache")


@pytest.fixture
def pipeline():
    return FakePipeline(SCORES)


@pytest.fixture
def classifier(manager, pipeline):
    make_ready_model(manager)
    return Classifier(manager, max_input_chars=20, pipeline_factory=lambda path: pipeline)


def test_valid_text_is_classified(classifier, pipeline):
    result = classifier.classify("  A close game.  ", MODEL.id, LABELS)

    assert result.input_text == "A close game."
    assert result.candidate_labels == tuple(LABELS)
    assert result.model_id == MODEL.id
    assert result.model_name == MODEL.name
    assert result.model_revision == MODEL.revision
    assert result.latency_ms >= 0
    assert pipeline.calls[0]["multi_label"] is False


def test_results_are_ranked_by_descending_score(classifier):
    result = classifier.classify("A close game.", MODEL.id, LABELS)

    assert [item.label for item in result.ranked_predictions] == ["sports", "finance", "politics"]
    scores = [item.score for item in result.ranked_predictions]
    assert scores == sorted(scores, reverse=True)


def test_predicted_class_is_the_highest_scoring_label(classifier):
    result = classifier.classify("A close game.", MODEL.id, LABELS)

    assert result.predicted_class == result.ranked_predictions[0].label == "sports"
    assert result.confidence == result.ranked_predictions[0].score == 0.9


def test_each_request_has_a_unique_id_and_utc_timestamp(classifier):
    first = classifier.classify("A close game.", MODEL.id, LABELS)
    second = classifier.classify("A close game.", MODEL.id, LABELS)

    assert first.request_id != second.request_id
    assert uuid.UUID(first.request_id).version == 4
    assert datetime.strptime(first.timestamp, TIMESTAMP_FORMAT)


@pytest.mark.parametrize("text", ["", "   ", "\n\t", None])
def test_empty_text_is_rejected(classifier, pipeline, text):
    with pytest.raises(InvalidInputError, match="Enter some text"):
        classifier.classify(text, MODEL.id, LABELS)

    assert pipeline.calls == []


def test_text_at_the_limit_is_accepted(classifier):
    assert classifier.classify("x" * 20, MODEL.id, LABELS).input_text == "x" * 20


def test_text_over_the_limit_is_rejected_with_both_lengths(classifier, pipeline):
    with pytest.raises(InvalidInputError, match=r"21 characters.*limit is 20"):
        classifier.classify("x" * 21, MODEL.id, LABELS)

    assert pipeline.calls == []


@pytest.mark.parametrize("labels", [[], ["sports"], ["sports", "Sports"], ["sports", ""]])
def test_invalid_labels_are_rejected(classifier, pipeline, labels):
    with pytest.raises(LabelValidationError):
        classifier.classify("A close game.", MODEL.id, labels)

    assert pipeline.calls == []


def test_model_that_is_not_ready_is_rejected(manager, pipeline):
    classifier = Classifier(manager, 20, pipeline_factory=lambda path: pipeline)

    with pytest.raises(ModelNotReadyError, match="Admin page"):
        classifier.classify("A close game.", MODEL.id, LABELS)


def test_model_without_pytorch_backend_is_rejected(manager):
    make_ready_model(manager)
    not_pytorch = FakePipeline(SCORES)
    not_pytorch.model = object()
    classifier = Classifier(manager, 20, pipeline_factory=lambda path: not_pytorch)

    with pytest.raises(ClassificationError, match="PyTorch"):
        classifier.classify("A close game.", MODEL.id, LABELS)


def test_pipeline_is_reused_for_the_same_model(manager, pipeline):
    make_ready_model(manager)
    created = []

    def factory(path):
        created.append(path)
        return pipeline

    classifier = Classifier(manager, 20, pipeline_factory=factory)
    classifier.classify("One.", MODEL.id, LABELS)
    classifier.classify("Two.", MODEL.id, LABELS)

    assert len(created) == 1


def test_only_one_pipeline_is_kept_in_memory(manager):
    make_ready_model(manager)
    make_ready_model(manager, OTHER.id, OTHER.revision)
    created = []

    def factory(path):
        created.append(path)
        return FakePipeline(SCORES)

    classifier = Classifier(manager, 20, pipeline_factory=factory)
    classifier.classify("One.", MODEL.id, LABELS)
    first_pipeline = classifier._pipeline
    classifier.classify("Two.", OTHER.id, LABELS)

    assert len(created) == 2
    assert classifier._pipeline is not first_pipeline
    assert classifier._loaded_model_id == OTHER.id


@pytest.mark.parametrize("failure", [RuntimeError, MemoryError, OSError, ValueError])
def test_inference_failures_give_a_clear_error(manager, failure):
    make_ready_model(manager)

    def fail(path):
        raise failure("internal detail hf_secretvalue123")

    classifier = Classifier(manager, 20, pipeline_factory=fail)

    with pytest.raises(ClassificationError, match="could not classify") as error:
        classifier.classify("A close game.", MODEL.id, LABELS)

    assert "internal detail" not in str(error.value)
