"""Shared test doubles and fixtures."""

import json

import torch

from app.services.classifier import ClassificationResult, RankedLabel
from app.services.model_manager import COMPLETION_MARKER, REQUIRED_MODEL_FILES, ModelConfig

MODEL = ModelConfig("org/model", "Test Model", "zero-shot-classification", "abc123")


class FakePipeline:
    """Gives each label a fixed score, returned unsorted to prove the code does the ranking."""

    def __init__(self, scores=None):
        self.model = torch.nn.Linear(1, 1)
        self.scores = scores or {}
        self.calls = []

    def __call__(self, text, candidate_labels, multi_label):
        self.calls.append(
            {"text": text, "candidate_labels": candidate_labels, "multi_label": multi_label}
        )
        return {
            "sequence": text,
            "labels": candidate_labels,
            "scores": [self.scores.get(label, 0.01) for label in candidate_labels],
        }


def make_ready_model(manager, model_id=MODEL.id, revision=MODEL.revision):
    path = manager.model_path(model_id)
    path.mkdir(parents=True)
    for name in REQUIRED_MODEL_FILES:
        (path / name).write_text("{}")
    (path / COMPLETION_MARKER).write_text(
        json.dumps(
            {"model_id": model_id, "revision": revision, "downloaded_at": "2026-11-01T00:00:00Z"}
        )
    )
    return path


def make_result(
    timestamp="2026-10-31T12:00:00Z", request_id="6a6d6a85-9dbb-4f3e-9f89-5d8e4e8c1b2a"
):
    return ClassificationResult(
        request_id=request_id,
        timestamp=timestamp,
        input_text="The team won the championship after a very close game.",
        predicted_class="sports",
        confidence=0.96,
        candidate_labels=("sports", "finance"),
        ranked_predictions=(RankedLabel("sports", 0.96), RankedLabel("finance", 0.04)),
        model_name="Test Model",
        model_id="org/model",
        model_revision="abc123",
        latency_ms=12.34,
    )
