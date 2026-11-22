"""Export real Langfuse trace summaries for the evaluation report (task T038).

    uv run python scripts/export_langfuse_traces.py                 # traces of the latest eval run
    uv run python scripts/export_langfuse_traces.py --trace-id <id>  # specific traces

Reads the observations of each trace through the Langfuse Observations API v2 (the traces
endpoints are outdated in Langfuse 5; see the official Langfuse skill, references/cli.md) and
writes reports/langfuse_traces_summary.json.

Honesty rules: only values returned by the API are written. A trace that cannot be fetched is
listed under "missing" with the reason; nothing is estimated or filled in.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.errors import safe_error_message  # noqa: E402

RESULTS = ROOT / "evaluation" / "results" / "latest.json"
OUT = ROOT / "reports" / "langfuse_traces_summary.json"
FIELD_SETS = ("core,basic,time,model,usage,metrics", "core,basic,usage")


def _get(obj: Any, name: str) -> Any:
    return getattr(obj, name, None)


def fetch_observations(client: Any, trace_id: str) -> list[Any]:
    last_error: Exception | None = None
    for fields in FIELD_SETS:
        try:
            rows, cursor = [], None
            while True:
                page = client.api.observations.get_many(
                    trace_id=trace_id, fields=fields, limit=100, cursor=cursor
                )
                rows.extend(page.data)
                cursor = _get(page.meta, "cursor")
                if not cursor:
                    return rows
        except Exception as exc:  # noqa: BLE001 - try the smaller field set, then give up
            last_error = exc
    raise last_error  # type: ignore[misc]


def _seconds(obs: Any) -> float | None:
    if _get(obs, "latency") is not None:
        return float(obs.latency)
    start, end = _get(obs, "start_time"), _get(obs, "end_time")
    if start and end:
        return (end - start).total_seconds()
    return None


def summarise_trace(trace_id: str, observations: list[Any]) -> dict[str, Any]:
    roots = [o for o in observations if not _get(o, "parent_observation_id")]
    root = roots[0] if roots else None
    types = Counter(str(_get(o, "type")) for o in observations)
    levels = Counter(str(_get(o, "level")) for o in observations)
    generations = [o for o in observations if str(_get(o, "type")).upper() == "GENERATION"]
    usage = Counter()
    for g in generations:
        for key, value in (_get(g, "usage_details") or {}).items():
            if isinstance(value, int | float):
                usage[key] += value
    costs = [_get(o, "total_cost") for o in observations if _get(o, "total_cost") is not None]
    names = Counter(str(_get(o, "name")) for o in observations)
    step_latency: dict[str, list[float]] = {}
    for o in observations:
        seconds = _seconds(o)
        if seconds is not None:
            step_latency.setdefault(str(_get(o, "type")), []).append(seconds)
    return {
        "trace_id": trace_id,
        "trace_name": _get(root, "trace_name") or _get(root, "name"),
        "session_id": _get(root, "session_id"),
        "start_time": str(_get(root, "start_time")) if root else None,
        "root_latency_s": round(_seconds(root), 3) if root and _seconds(root) is not None else None,
        "observation_count": len(observations),
        "observations_by_type": dict(types),
        "step_latency_s_by_type": {
            t: {"n": len(v), "total": round(sum(v), 3), "max": round(max(v), 3)}
            for t, v in step_latency.items()
        },
        "levels": dict(levels),
        "error_messages": [
            _get(o, "status_message") for o in observations if str(_get(o, "level")) == "ERROR"
        ],
        "has_retriever_observation": names.get("retrieve", 0) > 0,
        "has_guardrail_observation": names.get("citation-guardrail", 0) > 0,
        "generation_count": len(generations),
        "models": sorted({str(_get(g, "model")) for g in generations if _get(g, "model")}),
        "usage_details": dict(usage),
        "tokens_reported": bool(usage),
        "model_reported": any(_get(g, "model") for g in generations),
        "total_cost_usd": round(sum(costs), 6) if costs else None,
        "cost_reported": bool(costs),
    }


def aggregate(traces: list[dict[str, Any]]) -> dict[str, Any]:
    if not traces:
        return {"traces": 0}
    latencies = [t["root_latency_s"] for t in traces if t["root_latency_s"] is not None]
    tokens = [t["usage_details"].get("total") for t in traces if t["usage_details"].get("total")]
    costs = [t["total_cost_usd"] for t in traces if t["total_cost_usd"] is not None]
    return {
        "traces": len(traces),
        "with_error_level": sum("ERROR" in t["levels"] for t in traces),
        "with_retriever_observation": sum(t["has_retriever_observation"] for t in traces),
        "with_guardrail_observation": sum(t["has_guardrail_observation"] for t in traces),
        "with_token_usage": sum(t["tokens_reported"] for t in traces),
        "with_model_name": sum(t["model_reported"] for t in traces),
        "latency_s": {
            "median": round(statistics.median(latencies), 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
        },
        "total_tokens": sum(tokens) if tokens else None,
        "total_cost_usd": round(sum(costs), 6) if costs else None,
        "models": sorted({m for t in traces for m in t["models"]}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Langfuse trace summaries.")
    parser.add_argument("--trace-id", nargs="*", help="trace ids (default: latest eval run)")
    parser.add_argument("--wait", type=float, default=20, help="seconds to wait for ingestion")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.tracing_configured:
        print("Langfuse keys are not configured in .env; nothing to export.")
        return 2

    run_id = None
    trace_ids = args.trace_id or []
    if not trace_ids:
        if not RESULTS.exists():
            print(f"{RESULTS} not found. Run scripts/run_evaluation.py first, or pass --trace-id.")
            return 2
        run = json.loads(RESULTS.read_text(encoding="utf-8"))
        run_id = run["run_id"]
        trace_ids = [i["trace_id"] for i in run["items"] if i.get("trace_id")]
        if not trace_ids:
            print("The latest evaluation run has no trace ids (tracing was off).")
            return 2

    from langfuse import Langfuse

    client = Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
        tracing_enabled=False,
    )

    traces, missing = [], []
    deadline = time.monotonic() + args.wait
    pending = list(trace_ids)
    while pending:
        still = []
        for tid in pending:
            try:
                observations = fetch_observations(client, tid)
            except Exception as exc:  # noqa: BLE001
                missing.append({"trace_id": tid, "reason": safe_error_message(exc)})
                continue
            if observations:
                traces.append(summarise_trace(tid, observations))
            else:
                still.append(tid)
        if still and time.monotonic() < deadline:
            time.sleep(5)  # Langfuse processes ingested events asynchronously
            pending = still
        else:
            missing.extend({"trace_id": t, "reason": "no observations returned"} for t in still)
            pending = []

    summary = {
        "source": "Langfuse Observations API v2 (langfuse Python SDK 5.2.0)",
        "langfuse_base_url": settings.langfuse_base_url,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "evaluation_run_id": run_id,
        "requested": len(trace_ids),
        "aggregate": aggregate(traces),
        "traces": traces,
        "missing": missing,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Fetched {len(traces)}/{len(trace_ids)} traces; {len(missing)} missing. Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
