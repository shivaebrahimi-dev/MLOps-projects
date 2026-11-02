"""Candidate-label validation and label-set loading."""

import logging
import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.utils.errors import ConfigurationError, LabelValidationError
from app.utils.logging_config import log_event

logger = logging.getLogger(__name__)

MINIMUM_LABELS = 2
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class LabelSet:
    name: str
    labels: tuple[str, ...]


def sanitize_text(value: object) -> str:
    """Trim text and remove control characters, keeping tabs and line breaks."""
    return _CONTROL_CHARACTERS.sub("", value).strip() if isinstance(value, str) else ""


def validate_labels(labels: Iterable[object]) -> tuple[str, ...]:
    """Clean labels and reject empty values, duplicates, and sets that are too small."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for label in labels:
        text = sanitize_text(label)
        if not text:
            raise LabelValidationError("Labels cannot be empty.")
        if text.casefold() in seen:
            raise LabelValidationError(f"Label '{text}' is a duplicate.")
        seen.add(text.casefold())
        cleaned.append(text)
    if len(cleaned) < MINIMUM_LABELS:
        raise LabelValidationError(f"A label set needs at least {MINIMUM_LABELS} unique labels.")
    return tuple(cleaned)


def add_label(labels: Iterable[str], new_label: object) -> tuple[str, ...]:
    """Return the labels with one more, for a set that may still be too small to save."""
    text = sanitize_text(new_label)
    if not text:
        raise LabelValidationError("Labels cannot be empty.")
    current = tuple(labels)
    if text.casefold() in {label.casefold() for label in current}:
        raise LabelValidationError(f"Label '{text}' is a duplicate.")
    return (*current, text)


def remove_label(labels: Iterable[str], label: str) -> tuple[str, ...]:
    return tuple(item for item in labels if item != label)


def validate_set_name(name: object) -> str:
    text = name.strip() if isinstance(name, str) else ""
    if not text:
        raise LabelValidationError("A label set needs a name.")
    return text


class LabelManager:
    """Reads label sets from a YAML file. Each call reflects the current file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def label_sets(self) -> list[LabelSet]:
        return self._load()[1]

    def default_name(self) -> str:
        return self._load()[0]

    def get(self, name: str) -> LabelSet:
        for label_set in self.label_sets():
            if label_set.name.casefold() == name.strip().casefold():
                return label_set
        raise LabelValidationError(f"Label set '{name}' was not found.")

    def save(self, name: object, labels: Iterable[object], *, is_new: bool) -> LabelSet:
        """Validate and persist a label set. Existing sets are replaced, keeping their position."""
        saved = LabelSet(validate_set_name(name), validate_labels(labels))
        default, label_sets = self._load()
        folded = saved.name.casefold()
        position = next(
            (i for i, item in enumerate(label_sets) if item.name.casefold() == folded), None
        )
        if is_new and position is not None:
            raise LabelValidationError(f"A label set named '{saved.name}' already exists.")
        if position is None:
            label_sets.append(saved)
        else:
            label_sets[position] = saved
        self._write(default, label_sets)
        log_event(logger, "label_set_saved", label_set=saved.name, label_count=len(saved.labels))
        return saved

    def _write(self, default: str, label_sets: list[LabelSet]) -> None:
        document = {
            "default": default,
            "label_sets": [{"name": item.name, "labels": list(item.labels)} for item in label_sets],
        }
        temp_name = None
        try:
            # Replacing the file in one step means a failed save cannot corrupt it.
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent, suffix=".tmp", delete=False
            ) as temp_file:
                temp_name = temp_file.name
                yaml.safe_dump(document, temp_file, sort_keys=False, allow_unicode=True)
            os.replace(temp_name, self.path)
        except OSError:
            if temp_name is not None:
                Path(temp_name).unlink(missing_ok=True)
            raise ConfigurationError(
                f"The label set could not be saved to {self.path.name}. "
                "Check that the file is not read-only."
            ) from None

    def _load(self) -> tuple[str, list[LabelSet]]:
        file_name = self.path.name
        try:
            data = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        except OSError:
            raise ConfigurationError(
                f"Label configuration file {file_name} was not found."
            ) from None
        except yaml.YAMLError:
            raise ConfigurationError(
                f"Label configuration file {file_name} is not valid YAML."
            ) from None

        entries = data.get("label_sets") if isinstance(data, dict) else None
        if not isinstance(entries, list) or not entries:
            raise ConfigurationError(f"{file_name} must contain a non-empty `label_sets` list.")

        label_sets: list[LabelSet] = []
        seen: set[str] = set()
        try:
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("labels"), list):
                    raise LabelValidationError("Each label set needs a name and a list of labels.")
                name = validate_set_name(entry.get("name"))
                if name.casefold() in seen:
                    raise LabelValidationError(f"Label set '{name}' is a duplicate.")
                seen.add(name.casefold())
                label_sets.append(LabelSet(name, validate_labels(entry["labels"])))
        except LabelValidationError as error:
            raise ConfigurationError(f"{file_name} is invalid: {error}") from None

        default = data.get("default")
        if not isinstance(default, str) or default.strip().casefold() not in seen:
            raise ConfigurationError(f"{file_name} must set `default` to an existing label set.")
        return default.strip(), label_sets
