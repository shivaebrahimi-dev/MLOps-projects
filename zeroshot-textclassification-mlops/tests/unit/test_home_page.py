import pytest
from streamlit.testing.v1 import AppTest

from app.ui_state import _create_services
from app.utils.config import PROJECT_ROOT

HOME_PAGE = str(PROJECT_ROOT / "app" / "streamlit_app.py")


@pytest.fixture(autouse=True)
def clear_cached_services():
    _create_services.clear()
    yield
    _create_services.clear()


def test_home_page_shows_model_count(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test-access")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test-secret")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))

    page = AppTest.from_file(HOME_PAGE, default_timeout=60).run()

    assert not page.exception
    assert page.metric[0].value == "0 of 2"


def test_home_page_shows_configuration_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test-access")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test-secret")
    monkeypatch.setenv("MODELS_CONFIG", str(tmp_path / "missing.yaml"))

    page = AppTest.from_file(HOME_PAGE, default_timeout=60).run()

    assert not page.exception
    assert "not found" in page.error[0].value
