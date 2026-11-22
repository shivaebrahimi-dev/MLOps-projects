"""Tracing interface used by the rest of the app.

Two implementations:

* `NoopTracer`: does nothing. Used when tracing is off or keys are missing (FR-018), and in tests.
* `LangfuseTracer`: Langfuse Cloud via the Langfuse SDK and OpenInference instrumentors (added in
  the observability phase).

Other modules only use `Tracer`, `TraceHandle`, and `ObservationHandle`, so they never import
Langfuse directly (Constitution III).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any, Literal, Protocol

ObservationType = Literal["span", "agent", "tool", "chain", "retriever", "evaluator", "guardrail"]


class ObservationHandle(Protocol):
    def update(
        self,
        *,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        level: Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"] | None = None,
        status_message: str | None = None,
    ) -> None: ...


class TraceHandle(ObservationHandle, Protocol):
    trace_id: str | None
    trace_url: str | None


class Tracer(Protocol):
    enabled: bool
    description: str

    def question_trace(
        self, *, question: str, session_id: str, request_id: str, metadata: dict[str, Any]
    ) -> contextlib.AbstractContextManager[TraceHandle]: ...

    def observation(
        self,
        name: str,
        *,
        as_type: ObservationType = "span",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> contextlib.AbstractContextManager[ObservationHandle]: ...

    def flush(self) -> None: ...


class _NoopHandle:
    trace_id: str | None = None
    trace_url: str | None = None

    def update(self, **_: Any) -> None:
        return None


class NoopTracer:
    """Tracer that records nothing. The app behaves the same, minus the trace."""

    enabled = False

    def __init__(self, reason: str = "disabled") -> None:
        self.description = f"Langfuse: off ({reason})"

    @contextlib.contextmanager
    def question_trace(self, **_: Any) -> Iterator[TraceHandle]:
        yield _NoopHandle()

    @contextlib.contextmanager
    def observation(self, name: str, **_: Any) -> Iterator[ObservationHandle]:
        yield _NoopHandle()

    def flush(self) -> None:
        return None
