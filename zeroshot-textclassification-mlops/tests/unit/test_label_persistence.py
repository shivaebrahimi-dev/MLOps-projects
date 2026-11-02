import os

import pytest

from app.services.label_manager import LabelManager, add_label, remove_label
from app.utils.errors import ConfigurationError, LabelValidationError

START = "default: topics\nlabel_sets:\n  - {name: topics, labels: [sports, finance]}\n"


@pytest.fixture
def manager(tmp_path):
    path = tmp_path / "labels.yaml"
    path.write_text(START)
    return LabelManager(path)


def test_new_set_survives_a_reload(manager):
    manager.save("news", ["world", "local"], is_new=True)

    reloaded = LabelManager(manager.path)
    assert reloaded.get("news").labels == ("world", "local")
    assert reloaded.get("topics").labels == ("sports", "finance")


def test_existing_set_is_replaced_in_place(manager):
    manager.save("other", ["a", "b"], is_new=True)

    manager.save("TOPICS", ["sports", "finance", "health"], is_new=False)

    assert [item.name for item in manager.label_sets()] == ["TOPICS", "other"]
    assert manager.get("topics").labels == ("sports", "finance", "health")


def test_saving_keeps_the_default(manager):
    manager.save("news", ["world", "local"], is_new=True)

    assert manager.default_name() == "topics"


def test_new_set_cannot_reuse_a_name(manager):
    with pytest.raises(LabelValidationError, match="already exists"):
        manager.save("Topics", ["a", "b"], is_new=True)


@pytest.mark.parametrize(
    ("name", "labels", "message"),
    [
        ("news", ["only-one"], "at least 2"),
        ("news", ["a", "A"], "duplicate"),
        ("news", ["a", " "], "empty"),
        ("   ", ["a", "b"], "needs a name"),
    ],
)
def test_invalid_saves_are_refused_and_leave_the_file_unchanged(manager, name, labels, message):
    with pytest.raises(LabelValidationError, match=message):
        manager.save(name, labels, is_new=True)

    assert manager.path.read_text() == START


def test_failed_write_leaves_the_file_unchanged_and_no_temporary_file(manager, monkeypatch):
    def fail(source, target):
        raise PermissionError("read-only")

    monkeypatch.setattr(os, "replace", fail)

    with pytest.raises(ConfigurationError, match="could not be saved"):
        manager.save("news", ["world", "local"], is_new=True)

    assert manager.path.read_text() == START
    assert [path.name for path in manager.path.parent.iterdir()] == ["labels.yaml"]


def test_non_ascii_labels_are_kept(manager):
    manager.save("français", ["économie", "santé"], is_new=True)

    assert LabelManager(manager.path).get("français").labels == ("économie", "santé")


def test_add_label_trims_and_appends():
    assert add_label(["a"], "  b ") == ("a", "b")


@pytest.mark.parametrize(
    ("new_label", "message"), [("", "empty"), ("   ", "empty"), ("A", "duplicate")]
)
def test_add_label_rejects_invalid_values(new_label, message):
    with pytest.raises(LabelValidationError, match=message):
        add_label(["a"], new_label)


def test_draft_can_shrink_below_the_minimum_until_it_is_saved():
    assert add_label([], "first") == ("first",)
    assert remove_label(["a", "b"], "a") == ("b",)
