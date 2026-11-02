import pytest
from streamlit.testing.v1 import AppTest

import app.ui_state
from app.services.classifier import Classifier
from app.services.label_manager import LabelManager
from app.services.model_manager import ModelManager, ModelStatus
from app.ui_state import Services
from app.utils.config import PROJECT_ROOT, parse_settings
from tests.helpers import MODEL, make_ready_model

PAGE = str(PROJECT_ROOT / "app" / "pages" / "1_Admin.py")


def successful_download(model_id, revision, local_dir, patterns):
    local_dir.mkdir(parents=True)
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        (local_dir / name).write_text("{}")


def failing_download(model_id, revision, local_dir, patterns):
    raise ConnectionError("network down")


@pytest.fixture
def services(monkeypatch, tmp_path):
    def build(downloader=successful_download):
        labels_path = tmp_path / "labels.yaml"
        labels_path.write_text(
            "default: topics\nlabel_sets:\n  - {name: topics, labels: [sports, finance]}\n"
        )
        models = ModelManager([MODEL], tmp_path / "models", tmp_path / "cache", downloader)
        built = Services(
            settings=parse_settings({"MINIO_ACCESS_KEY": "a", "MINIO_SECRET_KEY": "b"}),
            models=models,
            labels=LabelManager(labels_path),
            classifier=Classifier(models, 100),
            storage=None,
        )
        monkeypatch.setattr(app.ui_state, "_create_services", lambda: built)
        return built

    return build


def open_page():
    return AppTest.from_file(PAGE, default_timeout=60).run()


def button(page, label):
    return next(item for item in page.button if item.label == label)


def table_cell(page, column):
    return page.table[0].value[column].iloc[0]


def download(page):
    page.multiselect[0].select(MODEL.id).run()
    return button(page, "Download selected").click().run()


def test_page_lists_models_locations_and_a_local_use_notice(services):
    built = services()

    page = open_page()

    assert not page.exception
    assert "no login" in page.warning[0].value
    assert table_cell(page, "Identifier") == MODEL.id
    assert table_cell(page, "Status") == "Not downloaded"
    assert str(built.models.model_dir) in page.text[0].value
    assert str(built.models.hf_home) in page.text[0].value


def test_download_button_needs_a_selection(services):
    services()

    assert button(open_page(), "Download selected").disabled


def test_download_shows_success_and_updates_the_table(services):
    built = services()

    page = download(open_page())

    assert not page.exception
    assert "ready" in page.success[0].value
    assert table_cell(page, "Status") == "Ready"
    assert built.models.status(MODEL.id).status is ModelStatus.READY


def test_ready_model_reports_it_is_already_available(services):
    built = services()
    make_ready_model(built.models)

    page = download(open_page())

    assert "already available" in page.info[0].value


def test_failed_download_shows_an_error_and_failed_status(services):
    services(failing_download)

    page = download(open_page())

    assert not page.exception
    assert "could not be downloaded" in page.error[0].value
    assert table_cell(page, "Status") == "Failed"
    assert "retry" in table_cell(page, "Details")


def test_refresh_reloads_status_from_disk(services):
    built = services()
    page = open_page()
    assert table_cell(page, "Status") == "Not downloaded"

    make_ready_model(built.models)
    page = button(page, "Refresh list").click().run()

    assert table_cell(page, "Status") == "Ready"
