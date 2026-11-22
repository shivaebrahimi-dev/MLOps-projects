"""Data model, as described in specs/001-agentic-rag/data-model.md."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

SAFE_ANSWER = "I could not find this in the uploaded knowledge."

AnswerStatus = Literal["answered", "partial", "not_found"]
ResultStatus = Literal["answered", "partial", "not_found", "error"]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    chunk_index: int
    text: str
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def chroma_metadata(self) -> dict[str, Any]:
        """Metadata in the flat, None-free form Chroma accepts."""
        meta = {"source": self.source, "chunk_index": self.chunk_index, **self.metadata}
        if self.page is not None:
            meta["page"] = self.page
        return {k: v for k, v in meta.items() if v is not None}


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int
    is_evidence: bool
    injection_flag: bool = False

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def source(self) -> str:
        return self.chunk.source

    def summary(self, snippet_chars: int = 160) -> dict[str, Any]:
        """Trace- and report-friendly view: ids, scores, and a short snippet, not the full text."""
        text = " ".join(self.chunk.text.split())
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page": self.chunk.page,
            "rank": self.rank,
            "score": round(self.score, 4),
            "is_evidence": self.is_evidence,
            "injection_flag": self.injection_flag,
            "snippet": text[:snippet_chars] + ("…" if len(text) > snippet_chars else ""),
        }


@dataclass(frozen=True)
class IngestResult:
    filename: str
    status: Literal["indexed", "rejected"]
    chunks: int
    message: str


class AnswerDraft(BaseModel):
    """Structured output of the Answer/Verifier task (contracts/answer.schema.json)."""

    status: AnswerStatus = Field(
        description="answered: fully supported; partial: only partly supported; "
        "not_found: no relevant evidence"
    )
    answer: str = Field(description="The answer, citing every claim inline as [source#n]")
    citations: list[str] = Field(
        default_factory=list, description="Chunk ids used, exactly as given in the evidence"
    )
    missing_evidence: list[str] = Field(
        default_factory=list, description="What the evidence does not cover (needed when partial)"
    )


@dataclass
class GuardrailReport:
    passed: bool = True
    attempts: int = 0
    reasons: list[str] = field(default_factory=list)
    fallback: bool = False


@dataclass
class AnswerResult:
    question: str
    status: ResultStatus
    answer: str
    request_id: str
    citations: list[RetrievedChunk] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    injection_flags: list[str] = field(default_factory=list)
    guardrail: GuardrailReport = field(default_factory=GuardrailReport)
    latency_s: float = 0.0
    token_usage: dict[str, int] | None = None
    model: str = ""
    trace_id: str | None = None
    trace_url: str | None = None
    error: str | None = None
    notices: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "status": self.status,
            "answer": self.answer,
            "request_id": self.request_id,
            "citations": [c.chunk_id for c in self.citations],
            "retrieved": [c.summary() for c in self.retrieved],
            "missing_evidence": self.missing_evidence,
            "injection_flags": self.injection_flags,
            "guardrail": {
                "passed": self.guardrail.passed,
                "attempts": self.guardrail.attempts,
                "reasons": self.guardrail.reasons,
                "fallback": self.guardrail.fallback,
            },
            "latency_s": round(self.latency_s, 3),
            "token_usage": self.token_usage,
            "model": self.model,
            "trace_id": self.trace_id,
            "trace_url": self.trace_url,
            "error": self.error,
        }
