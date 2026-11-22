"""Compare the baseline evaluation run with the experiment runs.

    uv run python scripts/compare_runs.py

Reads evaluation/results/latest.json (baseline) and evaluation/results/experiments/*.json, plus the
matching Langfuse exports when they exist, and writes evaluation/results/comparison.json.
Every value is copied from those files; a value a file does not contain is written as null.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation" / "results"
REPORTS = ROOT / "reports"


def _langfuse_for(run: dict[str, Any], baseline: bool) -> dict[str, Any] | None:
    path = (
        REPORTS / "langfuse_traces_summary.json"
        if baseline
        else REPORTS / "experiments" / f"langfuse_{run['run_id']}.json"
    )
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if data.get("evaluation_run_id") == run["run_id"] else None


def row(run: dict[str, Any], baseline: bool) -> dict[str, Any]:
    s, m = run["summary"], run["summary"]["metrics"]
    lf = _langfuse_for(run, baseline)
    return {
        "run_id": run["run_id"],
        "label": "baseline" if baseline else run.get("experiment"),
        "model": run["config"]["model"],
        "top_k": run["config"]["top_k"],
        "status_accuracy": s["status_accuracy"],
        "hit@k": m["hit@k"]["mean"],
        "recall@k": m["recall@k"]["mean"],
        "precision@k": m["precision@k"]["mean"],
        "mrr": m["mrr"]["mean"],
        "ndcg@k": m["ndcg@k"]["mean"],
        "citation_correctness": m["citation_correctness"]["mean"],
        "citation_completeness": m["citation_completeness"]["mean"],
        "answer_point_coverage": m["answer_point_coverage"]["mean"],
        "guardrail_fallbacks": s["guardrail"]["fallbacks"],
        "errors": s["errors"],
        "median_latency_s": s["latency_s"]["median"],
        "total_tokens": s["tokens"].get("total"),
        "langfuse_cost_usd": lf["aggregate"].get("total_cost_usd") if lf else None,
        "statuses": {i["id"]: i["status"] for i in run["items"]},
        "citations": {i["id"]: i["citations"] for i in run["items"]},
    }


def main() -> int:
    baseline_path = RESULTS / "latest.json"
    if not baseline_path.exists():
        print("No baseline: run scripts/run_evaluation.py first.")
        return 2
    runs = [row(json.loads(baseline_path.read_text(encoding="utf-8")), baseline=True)]
    for path in sorted((RESULTS / "experiments").glob("*.json")):
        runs.append(row(json.loads(path.read_text(encoding="utf-8")), baseline=False))

    (RESULTS / "comparison.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    cols = [
        "label",
        "model",
        "top_k",
        "status_accuracy",
        "hit@k",
        "precision@k",
        "citation_correctness",
        "citation_completeness",
        "answer_point_coverage",
        "median_latency_s",
        "total_tokens",
        "langfuse_cost_usd",
    ]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in runs:
        print("| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |")
    print(f"\nWrote {RESULTS / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
