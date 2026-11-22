"""Wiring: builds the embedder, knowledge base, tracer, and LLM factory from settings.

The UI and the scripts call `build_services()`; tests build the pieces themselves with fakes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from app.config import Settings
from app.crew import LLMFactory, default_llm_factory
from app.embeddings import Embedder, HashEmbedder, OpenAIEmbedder
from app.knowledge_base import KnowledgeBase
from app.observability import Tracer, init_tracing


def configure_runtime(settings: Settings) -> None:
    """Process-wide settings: logging, and CrewAI's own telemetry and cloud tracing switched off.

    Langfuse is this project's observability backend. CrewAI's anonymous usage telemetry and its
    cloud tracing prompt are disabled for privacy (research R8). `OTEL_SDK_DISABLED` is not used,
    because it would also disable the Langfuse exporter.
    """
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
    # Third-party libraries (CrewAI logs through the root logger) only show warnings; this
    # project's own modules log at LOG_LEVEL.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("app").setLevel(settings.log_level.upper())
    # Third-party request logs can include full URLs; keep them quiet.
    for noisy in ("httpx", "openai", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@dataclass
class Services:
    settings: Settings
    embedder: Embedder
    kb: KnowledgeBase
    tracer: Tracer
    llm_factory: LLMFactory | None
    problems: list[str]


def build_services(settings: Settings) -> Services:
    configure_runtime(settings)
    problems: list[str] = []
    if settings.has_openai_key:
        embedder: Embedder = OpenAIEmbedder(
            settings.embedding_model, settings.openai_api_key.get_secret_value()
        )
        llm_factory: LLMFactory | None = default_llm_factory(settings)
    else:
        problems.append("OPENAI_API_KEY is not set: indexing and answering are disabled.")
        embedder = HashEmbedder()  # lets the UI open and list the knowledge base
        llm_factory = None
    # One collection per embedding model, so vectors of different models never mix.
    collection = f"{settings.collection_name}-{embedder.name.replace('/', '-')}"
    kb = KnowledgeBase(settings.chroma_path, collection, embedder)
    tracer = init_tracing(settings)
    return Services(settings, embedder, kb, tracer, llm_factory, problems)
