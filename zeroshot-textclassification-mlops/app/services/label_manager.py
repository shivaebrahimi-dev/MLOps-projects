"""Candidate-label validation and label-set loading."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.utils.errors import ConfigurationError, LabelValidationError

MINIMUM_LABELS = 2


@dataclass(frozen=True)
class LabelSet:
    name: str
    labels: tuple[str, ...]


def validate_labels(labels: Iterable[object]) -> tuple[str, ...]:
    """Trim labels and reject empty values, duplicates, and sets that are too small."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for label in labels:
        text = label.strip() if isinstance(label, str) else ""
        if not text:
            raise LabelValidationError("Labels cannot be empty.")
        if text.casefold() in seen:
            raise LabelValidationError(f"Label '{text}' is a duplicate.")
        seen.add(text.casefold())
        cleaned.append(text)
    if len(cleaned) < MINIMUM_LABELS:
        raise LabelValidationError(f"A label set needs at least {MINIMUM_LABELS} unique labels.")
    return tuple(cleaned)


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
