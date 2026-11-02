import pytest
from streamlit.testing.v1 import AppTest

import app.ui_state
from app.services.classifier import Classifier
from app.services.label_manager import LabelManager
from app.services.model_manager import ModelManager
from app.ui_state import Services
from app.utils.config import PROJECT_ROOT, parse_settings
from app.utils.errors import StorageError
from tests.helpers import MODEL, FakePipeline, make_ready_model

PAGE = str(PROJECT_ROOT / "app" / "pages" / "2_Classify_Text.py")
SAMPLE = "The team won the championship."


class FakeStorage:
    def __init__(self, failure=None):
        self.failure = failure
        self.saved = []

    def save(self, result, session_id=None):
        if self.failure:
            raise self.failure
        self.saved.append((result, session_id))
        return type("Stored", (), {"bucket": "bucket", "object_path": "predictions/x.json"})()


@pytest.fixture
def setup(monkeypatch, tmp_path):
    labels_path = tmp_path / "labels.yaml"
    labels_path.write_text(
        "default: topics\nlabel_sets:\n  - {name: topics, labels: [sports, finance]}\n"
    )
    settings = parse_settings({"MINIO_ACCESS_KEY": "a", "MINIO_SECRET_KEY": "b"})
    models = ModelManager([MODEL], tmp_path / "models", tmp_path / "cache")
    storage = FakeStorage()
    pipeline = FakePipeline({"sports": 0.9, "finance": 0.1})
    services = Services(
        settings=settings,
        models=models,
        labels=LabelManager(labels_path),
        classifier=Classifier(models, 100, pipeline_factory=lambda path: pipeline),
        storage=storage,
    )
    monkeypatch.setattr(app.ui_state, "_create_services", lambda: services)
    return models, storage


def open_page():
    # The first render imports heavy libraries, which can exceed the default 3-second limit.
    return AppTest.from_file(PAGE, default_timeout=60).run()


def submit(text):
    page = open_page()
    page.text_area[0].set_value(text)
    return page.button[0].click().run()


def test_page_explains_when_no_model_is_ready(setup):
    page = open_page()

    assert not page.exception
    assert "Admin page" in page.warning[0].value
    assert not page.button


def test_result_is_shown_and_saved(setup):
    models, storage = setup
    make_ready_model(models)

    page = submit(SAMPLE)

    assert not page.exception
    assert page.metric[0].value == "sports"
    assert page.metric[1].value == "0.90"
    assert "predictions/x.json" in page.success[0].value
    assert page.text[0].value == SAMPLE
    assert "Request ID" in page.text[1].value
    assert len(storage.saved) == 1
    assert storage.saved[0][0].input_text == SAMPLE
    assert storage.saved[0][1]


def test_selected_model_and_label_set_are_the_ones_used(monkeypatch, tmp_path):
    labels_path = tmp_path / "labels.yaml"
    labels_path.write_text(
        "default: topics\n"
        "label_sets:\n"
        "  - {name: topics, labels: [sports, finance]}\n"
        "  - {name: moods, labels: [happy, sad]}\n"
    )
    other = type(MODEL)("org/other", "Other Model", MODEL.task, "def456")
    models = ModelManager([MODEL, other], tmp_path / "models", tmp_path / "cache")
    make_ready_model(models)
    make_ready_model(models, other.id, other.revision)
    loaded_paths = []

    def factory(path):
        loaded_paths.append(path)
        return FakePipeline({"happy": 0.8, "sad": 0.2})

    storage = FakeStorage()
    services = Services(
        settings=parse_settings({"MINIO_ACCESS_KEY": "a", "MINIO_SECRET_KEY": "b"}),
        models=models,
        labels=LabelManager(labels_path),
        classifier=Classifier(models, 100, pipeline_factory=factory),
        storage=storage,
    )
    monkeypatch.setattr(app.ui_state, "_create_services", lambda: services)

    # A browser sends all of a form's values together when it is submitted.
    page = open_page()
    page.text_area[0].set_value(SAMPLE)
    page.selectbox[0].select("Other Model")
    page.selectbox[1].select(page.selectbox[1].options[1])
    page = page.button[0].click().run()

    assert not page.exception
    result = storage.saved[0][0]
    assert result.model_id == other.id
    assert result.model_revision == other.revision
    assert result.candidate_labels == ("happy", "sad")
    assert loaded_paths == [models.model_path(other.id)]


def test_empty_text_shows_an_error_and_saves_nothing(setup):
    models, storage = setup
    make_ready_model(models)

    page = submit("   ")

    assert "Enter some text" in page.error[0].value
    assert not page.metric
    assert storage.saved == []


def test_text_over_the_limit_shows_an_error(setup):
    models, storage = setup
    make_ready_model(models)

    page = submit("x" * 101)

    assert "limit is 100" in page.error[0].value
    assert storage.saved == []


def test_storage_failure_still_shows_the_result(setup):
    models, storage = setup
    make_ready_model(models)
    storage.failure = StorageError("The result was not saved: MinIO could not be reached.")

    page = submit(SAMPLE)

    assert not page.exception
    assert page.metric[0].value == "sports"
    assert "not saved" in page.error[0].value
    assert "Request ID" in page.error[0].value
