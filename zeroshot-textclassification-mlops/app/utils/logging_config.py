"""Structured JSON logging that keeps credentials and input text out of the logs."""

import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any

REDACTED = "[REDACTED]"

_SENSITIVE_FIELD = re.compile(r"secret|token|password|credential|access_key|api_key", re.IGNORECASE)
_INPUT_FIELDS = {"text", "input_text"}
_SECRET_VALUE = re.compile(r"\bhf_[A-Za-z0-9]{8,}\b")
_FIELDS_ATTRIBUTE = "structured_fields"


def _clean_text(value: str) -> str:
    return _SECRET_VALUE.sub(REDACTED, value)


def _clean_fields(fields: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for name, value in fields.items():
        if name in _INPUT_FIELDS:
            continue
        if _SENSITIVE_FIELD.search(name):
            cleaned[name] = REDACTED
        elif isinstance(value, str):
            cleaned[name] = _clean_text(value)
        else:
            cleaned[name] = value
    return cleaned


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": _clean_text(record.getMessage()),
        }
        entry.update(_clean_fields(getattr(record, _FIELDS_ATTRIBUTE, {})))
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception_type"] = record.exc_info[0].__name__
        return json.dumps(entry, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Send JSON logs to stderr. Calling it again only updates the level."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    if not any(isinstance(handler.formatter, JsonFormatter) for handler in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)


def log_event(
    logger: logging.Logger, event: str, *, level: int = logging.INFO, **fields: Any
) -> None:
    """Log an event with structured metadata."""
    logger.log(level, event, extra={_FIELDS_ATTRIBUTE: fields})
