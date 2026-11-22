"""Application settings, read from environment variables and an optional `.env` file.

Empty values in `.env` fall back to the defaults below, so `.env.example` can list names only.
Secrets are `SecretStr`, so they never appear in `repr()`, logs, or tracebacks.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_VERSION = "0.1.0"
SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf")
MAX_QUESTION_CHARS = 1000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # OpenAI
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Langfuse
    langfuse_enabled: bool = True
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"

    # Knowledge base and retrieval
    chroma_dir: Path = Path(".chroma")
    collection_name: str = "knowledge"
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    top_k: int = Field(default=6, ge=1, le=20)
    min_relevance: float = Field(default=0.30, ge=-1.0, le=1.0)
    max_upload_mb: float = Field(default=10, gt=0)

    log_level: str = "INFO"

    @property
    def chroma_path(self) -> Path:
        """Absolute store path; relative paths are resolved against the project root."""
        path = self.chroma_dir
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def tracing_configured(self) -> bool:
        """True when tracing is switched on and both Langfuse keys are present."""
        return bool(
            self.langfuse_enabled
            and self.langfuse_public_key
            and self.langfuse_public_key.get_secret_value()
            and self.langfuse_secret_key
            and self.langfuse_secret_key.get_secret_value()
        )

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.get_secret_value())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Settings for the app. `RAG_IGNORE_ENV_FILE=1` skips `.env` (used by the UI test)."""
    if os.environ.get("RAG_IGNORE_ENV_FILE") == "1":
        return Settings(_env_file=None)
    return Settings()
