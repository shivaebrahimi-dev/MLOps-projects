"""Typed errors whose messages are safe to show to users."""


class ZeroShotError(Exception):
    """Base error. Its message must never contain secrets or stack traces."""

    @property
    def user_message(self) -> str:
        return str(self)


class ConfigurationError(ZeroShotError):
    """A setting or configuration file is missing or invalid."""


class InvalidInputError(ZeroShotError):
    """Text submitted for classification is not valid."""


class LabelValidationError(ZeroShotError):
    """A candidate label or label set is not valid."""


class ModelNotReadyError(ZeroShotError):
    """The selected model is not available for inference."""


class ClassificationError(ZeroShotError):
    """Inference could not be completed."""


class ModelDownloadError(ZeroShotError):
    """A model could not be downloaded."""


class ModelOperationError(ZeroShotError):
    """A model deletion or cache cleanup could not be completed."""


class ModelInUseError(ModelOperationError):
    """A model cannot be changed because a download is running."""


class StorageError(ZeroShotError):
    """A prediction could not be stored in MinIO."""
