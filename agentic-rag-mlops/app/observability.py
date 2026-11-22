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
import logging
from collections.abc import Iterator
from typing import Any, Literal, Protocol

from app.config import APP_VERSION, Settings
from app.errors import safe_error_message

log = logging.getLogger(__name__)

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


# -- Langfuse ------------------------------------------------------------------------------------


class _LangfuseHandle:
    def __init__(self, observation: Any, trace_id: str | None = None, trace_url: str | None = None):
        self._observation = observation
        self.trace_id = trace_id
        self.trace_url = trace_url

    def update(
        self,
        *,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        level: Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"] | None = None,
        status_message: str | None = None,
    ) -> None:
        fields = {
            "output": output,
            "metadata": metadata,
            "level": level,
            "status_message": status_message,
        }
        self._observation.update(**{k: v for k, v in fields.items() if v is not None})


def _baggage_safe(metadata: dict[str, Any]) -> dict[str, str]:
    """`propagate_attributes` metadata must be short strings."""
    return {str(k): str(v)[:200] for k, v in metadata.items() if v is not None}


class LangfuseTracer:
    """Langfuse Cloud tracing (research R8). CrewAI and OpenAI spans come from OpenInference."""

    enabled = True

    def __init__(self, client: Any, base_url: str, propagate: Any | None = None) -> None:
        if propagate is None:
            from langfuse import propagate_attributes as propagate
        self.client = client
        self._propagate = propagate
        self.description = f"Langfuse: on ({base_url})"

    @contextlib.contextmanager
    def question_trace(
        self, *, question: str, session_id: str, request_id: str, metadata: dict[str, Any]
    ) -> Iterator[TraceHandle]:
        with self._propagate(
            session_id=session_id,
            trace_name="rag-question",
            tags=["agentic-rag"],
            metadata=_baggage_safe({"request_id": request_id, **metadata}),
        ):
            with self.client.start_as_current_observation(
                name="rag-question",
                as_type="agent",
                input={"question": question, "request_id": request_id},
                metadata=metadata,
            ) as root:
                trace_id = self.client.get_current_trace_id()
                try:
                    trace_url = self.client.get_trace_url(trace_id=trace_id)
                except Exception:  # noqa: BLE001 - the link is a convenience only
                    trace_url = None
                yield _LangfuseHandle(root, trace_id=trace_id, trace_url=trace_url)

    @contextlib.contextmanager
    def observation(
        self,
        name: str,
        *,
        as_type: ObservationType = "span",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[ObservationHandle]:
        with self.client.start_as_current_observation(
            name=name, as_type=as_type, input=input, metadata=metadata
        ) as obs:
            yield _LangfuseHandle(obs)

    def flush(self) -> None:
        try:
            self.client.flush()
        except Exception:  # noqa: BLE001 - never fail a user request because export failed
            log.warning("Langfuse flush failed", exc_info=True)


_instrumented = False


def instrument_once() -> None:
    """Register the OpenInference CrewAI and OpenAI instrumentors (once per process)."""
    global _instrumented
    if _instrumented:
        return
    from openinference.instrumentation.crewai import CrewAIInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor

    CrewAIInstrumentor().instrument(skip_dep_check=True)
    OpenAIInstrumentor().instrument()
    _instrumented = True


def init_tracing(settings: Settings, client_factory: Any | None = None) -> Tracer:
    """Return a working tracer, or a `NoopTracer` that says why tracing is off (FR-018)."""
    if not settings.langfuse_enabled:
        return NoopTracer("LANGFUSE_ENABLED=false")
    if not settings.tracing_configured:
        return NoopTracer("Langfuse keys are not set")
    try:
        if client_factory is None:
            from langfuse import Langfuse as client_factory
        client = client_factory(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            base_url=settings.langfuse_base_url,
            release=APP_VERSION,
            environment="local",
        )
        if not client.auth_check():
            return NoopTracer("Langfuse rejected the keys; check them and LANGFUSE_BASE_URL")
    except Exception as exc:  # noqa: BLE001 - tracing must never stop the app
        return NoopTracer(f"Langfuse unavailable: {safe_error_message(exc)}")
    instrument_once()
    return LangfuseTracer(client, settings.langfuse_base_url)
