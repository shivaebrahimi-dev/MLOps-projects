"""Live check against the real OpenAI and Langfuse services (task T029).

Skipped by default. Run with:

    RUN_LIVE_TESTS=1 uv run pytest -m live -q          (PowerShell: $env:RUN_LIVE_TESTS="1")

Uses the keys in .env, a temporary knowledge base, and one sample question.
"""

from __future__ import annotations

import os

import pytest

from app.config import Settings
from app.crew import answer_question
from app.ingestion import ingest_files
from app.services import build_services
from tests.conftest import SAMPLE_DIR

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1"),
]


def test_real_question_is_answered_with_citations_and_traced(tmp_path):
    settings = Settings().model_copy(update={"chroma_dir": tmp_path / "chroma"})
    svc = build_services(settings)
    assert svc.llm_factory is not None, "OPENAI_API_KEY missing in .env"
    files = [(p.name, p.read_bytes()) for p in sorted(SAMPLE_DIR.iterdir())]
    assert all(r.status == "indexed" for r in ingest_files(files, svc.kb, settings, svc.tracer))

    result = answer_question(
        "Which MinIO bucket stores the Fashion-MNIST model files?",
        kb=svc.kb,
        settings=settings,
        llm_factory=svc.llm_factory,
        tracer=svc.tracer,
    )
    assert result.status == "answered", result.error
    assert any(c.source == "fashion-mnist-pipeline.md" for c in result.citations)
    assert "fashion-minst-bucket" in result.answer
    assert result.token_usage and result.token_usage["total"] > 0
    if settings.tracing_configured:
        assert svc.tracer.enabled, svc.tracer.description
        assert result.trace_id
        print(f"\nLangfuse trace id: {result.trace_id}\n{result.trace_url}")
