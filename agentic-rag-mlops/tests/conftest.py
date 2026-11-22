"""Shared fixtures. Every test runs offline, with no real keys and tracing off (Constitution V)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, Settings
from app.embeddings import HashEmbedder
from app.knowledge_base import KnowledgeBase

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample-knowledge"

_SECRET_VARS = (
    "OPENAI_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_HOST",
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove real credentials and switch off every telemetry channel for unit tests."""
    if os.environ.get("RUN_LIVE_TESTS") == "1":
        return
    for name in _SECRET_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings that ignore the developer's `.env` file and use a temporary store."""
    return Settings(
        _env_file=None,
        langfuse_enabled=False,
        chroma_dir=tmp_path / "chroma",
        chunk_size=400,
        chunk_overlap=60,
        top_k=4,
        min_relevance=0.15,
    )


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture
def kb(settings: Settings, embedder: HashEmbedder) -> KnowledgeBase:
    return KnowledgeBase(settings.chroma_path, "test-knowledge", embedder)
