"""Health check of the Streamlit UI: it starts, shows both tabs, and explains missing setup."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from app.config import PROJECT_ROOT, get_settings


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_IGNORE_ENV_FILE", "1")  # never read the developer's real .env
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    get_settings.cache_clear()
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "ui.py"), default_timeout=60)
    at.run()
    yield at
    get_settings.cache_clear()


def test_ui_starts_with_two_tabs_and_no_exceptions(app):
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Knowledge", "Ask"]
    assert app.title[0].value == "Agentic RAG Knowledge Assistant"


def test_ui_explains_missing_keys_and_tracing_off(app):
    assert any("OPENAI_API_KEY is not set" in e.value for e in app.error)
    assert "Langfuse: off" in app.caption[0].value


def test_ask_tab_asks_for_uploads_when_knowledge_is_empty(app):
    assert any("Upload files in the Knowledge tab first" in i.value for i in app.info)
