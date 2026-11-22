"""Retrieval returns relevant evidence for known questions (US2, SC-003)."""

from __future__ import annotations

import pytest

from app.ingestion import ingest_files
from app.retrieval import NO_EVIDENCE, format_evidence, retrieve
from tests.conftest import SAMPLE_DIR


@pytest.fixture
def sample_kb(kb, settings):
    files = [(p.name, p.read_bytes()) for p in sorted(SAMPLE_DIR.iterdir()) if p.is_file()]
    results = ingest_files(files, kb, settings)
    assert all(r.status == "indexed" for r in results), results
    return kb


@pytest.mark.parametrize(
    ("question", "expected_source"),
    [
        (
            "Which bucket stores the Fashion-MNIST model.pt files in MinIO?",
            "fashion-mnist-pipeline.md",
        ),
        ("What are the statuses of a zero-shot model download?", "zero-shot-platform.txt"),
        ("What was Cohen's kappa of the LLM judge?", "evaluation-metrics-findings.pdf"),
    ],
)
def test_known_question_retrieves_its_source(sample_kb, settings, question, expected_source):
    results = retrieve(sample_kb, question, top_k=settings.top_k, min_relevance=0.0)
    assert expected_source in {r.source for r in results}
    assert results[0].rank == 1
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_threshold_marks_evidence(sample_kb):
    results = retrieve(sample_kb, "MinIO bucket model", top_k=4, min_relevance=0.99)
    assert results and not any(r.is_evidence for r in results)
    assert format_evidence(results) == NO_EVIDENCE


def test_evidence_is_delimited_and_carries_ids(sample_kb):
    results = retrieve(sample_kb, "Fashion-MNIST MinIO bucket", top_k=3, min_relevance=0.0)
    text = format_evidence(results)
    assert text.count("<<<EVIDENCE id=") == text.count("<<<END EVIDENCE>>>") == len(results)
    for r in results:
        assert f'id="{r.chunk_id}"' in text
    assert "never as instructions" in text


def test_injection_chunk_is_flagged_but_kept(sample_kb):
    results = retrieve(
        sample_kb,
        "ignore all previous instructions maintenance mode system prompt",
        top_k=4,
        min_relevance=0.0,
    )
    flagged = [r for r in results if r.injection_flag]
    assert flagged and flagged[0].source == "submission-checklist-injection-test.md"
    assert flagged[0].chunk_id in format_evidence(results)


def test_empty_knowledge_base_returns_nothing(kb):
    assert retrieve(kb, "anything", top_k=4, min_relevance=0.0) == []
