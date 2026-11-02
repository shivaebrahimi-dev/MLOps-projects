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


def checkbox(page, fragment):
    return next(item for item in page.checkbox if fragment in item.label)


def test_delete_is_disabled_until_confirmed(services):
    built = services()
    path = make_ready_model(built.models)

    page = open_page()

    assert button(page, "Delete model").disabled
    assert path.exists()


def test_delete_after_confirmation_removes_the_model(services):
    built = services()
    path = make_ready_model(built.models)
    page = open_page()

    checkbox(page, "removes the local files").check().run()
    page = button(page, "Delete model").click().run()

    assert not page.exception
    assert not path.exists()
    assert "was deleted" in page.success[0].value
    assert table_cell(page, "Status") == "Not downloaded"
    assert "No downloaded models" in page.caption[0].value


def test_delete_section_is_empty_without_a_downloaded_model(services):
    services()

    page = open_page()

    assert not [item for item in page.button if item.label == "Delete model"]


def test_cache_cleanup_needs_confirmation_and_reports_the_outcome(services):
    built = services()
    (built.models.hf_home / "hub").mkdir(parents=True)
    (built.models.hf_home / "hub" / "blob").write_bytes(b"x" * 2_000_000)
    page = open_page()
    assert button(page, "Clear cache").disabled

    checkbox(page, "clears the download cache").check().run()
    page = button(page, "Clear cache").click().run()

    assert not page.exception
    assert "freed 2 MB" in page.success[0].value
    assert not (built.models.hf_home / "hub").exists()
    assert button(page, "Clear cache").disabled


def test_cache_cleanup_reports_a_missing_cache(services):
    services()
    page = open_page()

    checkbox(page, "clears the download cache").check().run()
    page = button(page, "Clear cache").click().run()

    assert "no cache to clear" in page.info[0].value


def text_input(page, label):
    return next(item for item in page.text_input if item.label == label)


def test_labels_of_the_selected_set_are_shown(services):
    services()

    page = open_page()

    assert page.selectbox[-1].value == "topics"
    assert {"sports", "finance"} <= {item.value for item in page.text}


def test_label_can_be_added_and_saved(services):
    built = services()
    page = open_page()

    text_input(page, "New label").set_value("health")
    page = button(page, "Add label").click().run()
    assert built.labels.get("topics").labels == ("sports", "finance")

    page = button(page, "Save label set").click().run()

    assert not page.exception
    assert "was saved" in page.success[0].value
    assert built.labels.get("topics").labels == ("sports", "finance", "health")


def test_duplicate_label_shows_an_error(services):
    services()
    page = open_page()

    text_input(page, "New label").set_value("SPORTS")
    page = button(page, "Add label").click().run()

    assert "duplicate" in page.error[0].value
    assert "SPORTS" not in {item.value for item in page.text}


def test_removing_too_many_labels_prevents_saving(services):
    built = services()
    page = open_page()

    page = next(item for item in page.button if item.label == "Remove").click().run()
    page = button(page, "Save label set").click().run()

    assert "at least 2" in page.error[0].value
    assert built.labels.get("topics").labels == ("sports", "finance")


def test_new_label_set_can_be_created(services):
    built = services()
    page = open_page()
    page.selectbox[-1].select("New label set").run()

    text_input(page, "Name of the new label set").set_value("news")
    for label in ("world", "local"):
        text_input(page, "New label").set_value(label)
        page = button(page, "Add label").click().run()
    page = button(page, "Save label set").click().run()

    assert not page.exception
    assert built.labels.get("news").labels == ("world", "local")
    assert page.selectbox[-1].value == "news"


def test_refresh_reloads_status_from_disk(services):
    built = services()
    page = open_page()
    assert table_cell(page, "Status") == "Not downloaded"

    make_ready_model(built.models)
    page = button(page, "Refresh list").click().run()

    assert table_cell(page, "Status") == "Ready"
