"""Exceptions with messages that are safe to show to the user."""

from __future__ import annotations


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
