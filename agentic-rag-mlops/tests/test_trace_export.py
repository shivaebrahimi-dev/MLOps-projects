"""The Langfuse export summarises only what the API returned (no invented values)."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from types import SimpleNamespace

from app.config import PROJECT_ROOT

spec = importlib.util.spec_from_file_location(
    "export_langfuse_traces", PROJECT_ROOT / "scripts" / "export_langfuse_traces.py"
)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)

T0 = datetime(2026, 11, 22, 12, 0, 0)


def obs(name, type_, parent=None, level="DEFAULT", seconds=1.0, **extra):
    return SimpleNamespace(
        name=name,
        type=type_,
        parent_observation_id=parent,
        level=level,
        start_time=T0,
        end_time=T0 + timedelta(seconds=seconds),
        latency=None,
        trace_name="rag-question" if parent is None else None,
        session_id="s1",
        status_message=extra.get("status_message"),
        model=extra.get("model"),
        usage_details=extra.get("usage_details"),
        total_cost=extra.get("total_cost"),
    )


def test_summary_counts_types_tokens_and_cost():
    observations = [
        obs("rag-question", "AGENT", seconds=6.5),
        obs("retrieve", "RETRIEVER", parent="root"),
        obs("citation-guardrail", "GUARDRAIL", parent="root"),
        obs(
            "ChatCompletion",
            "GENERATION",
            parent="root",
            model="gpt-4o-mini",
            usage_details={"input": 900, "output": 100, "total": 1000},
            total_cost=0.0002,
        ),
        obs(
            "ChatCompletion",
            "GENERATION",
            parent="root",
            model="gpt-4o-mini",
            usage_details={"input": 50, "output": 10, "total": 60},
            total_cost=0.00001,
        ),
    ]
    s = export.summarise_trace("t1", observations)
    assert s["trace_name"] == "rag-question" and s["root_latency_s"] == 6.5
    assert s["observations_by_type"]["GENERATION"] == 2
    assert s["usage_details"] == {"input": 950, "output": 110, "total": 1060}
    assert s["tokens_reported"] and s["model_reported"] and s["models"] == ["gpt-4o-mini"]
    assert s["total_cost_usd"] == 0.00021
    assert s["has_retriever_observation"] and s["has_guardrail_observation"]
    assert s["step_latency_s_by_type"]["GENERATION"] == {"n": 2, "total": 2.0, "max": 1.0}


def test_missing_values_stay_missing():
    s = export.summarise_trace(
        "t2", [obs("rag-question", "AGENT", level="ERROR", status_message="RuntimeError: down")]
    )
    assert s["tokens_reported"] is False and s["usage_details"] == {}
    assert s["total_cost_usd"] is None and s["cost_reported"] is False
    assert s["error_messages"] == ["RuntimeError: down"]
    agg = export.aggregate([s])
    assert agg["total_tokens"] is None and agg["total_cost_usd"] is None
    assert agg["with_error_level"] == 1
