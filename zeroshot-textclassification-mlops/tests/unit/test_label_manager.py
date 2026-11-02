import pytest

from app.services.label_manager import LabelManager, validate_labels
from app.utils.config import PROJECT_ROOT
from app.utils.errors import ConfigurationError, LabelValidationError


def write_labels(tmp_path, text):
    path = tmp_path / "labels.yaml"
    path.write_text(text)
    return LabelManager(path)


def test_labels_are_trimmed():
    assert validate_labels(["  sports ", "finance"]) == ("sports", "finance")


def test_control_characters_are_removed_from_labels():
    assert validate_labels(["spo\x00rts", "fin\x1bance"]) == ("sports", "finance")


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ([], "at least 2"),
        (["sports"], "at least 2"),
        (["sports", ""], "empty"),
        (["sports", "   "], "empty"),
        (["sports", None], "empty"),
        (["sports", "sports"], "duplicate"),
        (["sports", "Sports"], "duplicate"),
        (["sports", "  SPORTS  "], "duplicate"),
    ],
)
def test_invalid_labels_are_rejected(labels, message):
    with pytest.raises(LabelValidationError, match=message):
        validate_labels(labels)


def test_default_configuration_loads():
    manager = LabelManager(PROJECT_ROOT / "configs" / "labels.yaml")

    assert manager.default_name() == "general-topics"
    assert manager.get("general-topics").labels == (
        "sports",
        "politics",
        "technology",
        "finance",
        "health",
        "education",
    )


def test_label_set_lookup_ignores_case(tmp_path):
    manager = write_labels(
        tmp_path, "default: Topics\nlabel_sets:\n  - {name: Topics, labels: [a, b]}\n"
    )

    assert manager.get("topics").name == "Topics"


def test_unknown_label_set_is_reported(tmp_path):
    manager = write_labels(
        tmp_path, "default: topics\nlabel_sets:\n  - {name: topics, labels: [a, b]}\n"
    )

    with pytest.raises(LabelValidationError, match="not found"):
        manager.get("other")


def test_file_changes_are_picked_up(tmp_path):
    manager = write_labels(
        tmp_path, "default: first\nlabel_sets:\n  - {name: first, labels: [a, b]}\n"
    )
    assert [label_set.name for label_set in manager.label_sets()] == ["first"]

    manager.path.write_text("default: second\nlabel_sets:\n  - {name: second, labels: [a, b]}\n")

    assert [label_set.name for label_set in manager.label_sets()] == ["second"]


@pytest.mark.parametrize(
    "text",
    [
        "label_sets: [",
        "other: 1",
        "default: a\nlabel_sets: []",
        "default: missing\nlabel_sets:\n  - {name: a, labels: [x, y]}",
        "label_sets:\n  - {name: a, labels: [x, y]}",
        "default: a\nlabel_sets:\n  - {name: a, labels: [x]}",
        "default: a\nlabel_sets:\n  - {name: a}",
        "default: a\nlabel_sets:\n  - {name: a, labels: [x, y]}\n  - {name: A, labels: [x, y]}",
    ],
)
def test_invalid_label_files_are_rejected(tmp_path, text):
    with pytest.raises(ConfigurationError, match="labels.yaml"):
        write_labels(tmp_path, text).label_sets()


def test_missing_label_file_is_reported(tmp_path):
    with pytest.raises(ConfigurationError, match="not found"):
        LabelManager(tmp_path / "missing.yaml").label_sets()
