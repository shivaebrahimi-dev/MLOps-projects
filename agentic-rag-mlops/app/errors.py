"""Exceptions with messages that are safe to show to the user."""

from __future__ import annotations

import re


class UploadRejected(Exception):
    """An uploaded file cannot be indexed. `reason` is shown in the UI."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class UnsupportedFileType(UploadRejected):
    pass


class EmptyFile(UploadRejected):
    pass


class NoExtractableText(UploadRejected):
    pass


class FileTooLarge(UploadRejected):
    pass


class InvalidQuestion(ValueError):
    """The question is empty after trimming."""


class ConfigurationError(RuntimeError):
    """A required setting (for example the OpenAI key) is missing."""


_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-*]{6,}"),  # OpenAI keys, also partly masked ones
    re.compile(r"(?:pk|sk)-lf-[A-Za-z0-9\-]{6,}"),  # Langfuse keys
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{10,}"),
]


def redact(text: str) -> str:
    """Remove anything that looks like an API key."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + "***", text)
    return text


def safe_error_message(exc: BaseException, limit: int = 300) -> str:
    """A short, secret-free description of an exception, safe for the UI, logs, and traces."""
    message = redact(str(exc)) or "no details"
    if len(message) > limit:
        message = message[:limit] + "…"
    return f"{type(exc).__name__}: {message}"
