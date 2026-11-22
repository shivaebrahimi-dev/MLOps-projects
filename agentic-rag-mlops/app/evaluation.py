"""Evaluation metrics (contracts/evaluation-dataset.md), as pure functions.

Relevance is judged at document level: a retrieved chunk is relevant when its source is one of the
item's expected sources. Functions return None when a metric does not apply, so summaries never
average a value that was not measured.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_dataset(questions: Path, expected: Path) -> list[dict[str, Any]]:
    """Join questions and the rubric on `id`; every question must have a rubric row."""
    rubric = {row["id"]: row for row in load_jsonl(expected)}
    items = []
    for q in load_jsonl(questions):
        if q["id"] not in rubric:
            raise ValueError(f"No expected answer for {q['id']}")
        items.append({**q, **rubric[q["id"]]})
    return items


# -- retrieval -----------------------------------------------------------------------------------


def _relevance(sources: Sequence[str], expected: Iterable[str]) -> list[int]:
    expected = set(expected)
    return [1 if s in expected else 0 for s in sources]


def hit_at_k(sources: Sequence[str], expected: Sequence[str]) -> float | None:
    if not expected:
        return None
    return 1.0 if any(_relevance(sources, expected)) else 0.0


def recall_at_k(sources: Sequence[str], expected: Sequence[str]) -> float | None:
    if not expected:
        return None
    return len(set(sources) & set(expected)) / len(set(expected))


def precision_at_k(sources: Sequence[str], expected: Sequence[str]) -> float | None:
    if not expected or not sources:
        return None if not expected else 0.0
    return sum(_relevance(sources, expected)) / len(sources)


def mrr(sources: Sequence[str], expected: Sequence[str]) -> float | None:
    if not expected:
        return None
    for rank, rel in enumerate(_relevance(sources, expected), start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(sources: Sequence[str], expected: Sequence[str], n_relevant: int) -> float | None:
    """Binary nDCG. `n_relevant` = relevant chunks that exist in the knowledge base."""
    if not expected:
        return None
    rels = _relevance(sources, expected)
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
    ideal = min(len(sources), n_relevant)
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg else 0.0


# -- citations and answers -----------------------------------------------------------------------


def _source_of(chunk_id: str) -> str:
    return chunk_id.rsplit("#", 1)[0]


def citation_correctness(
    cited: Sequence[str], retrieved_ids: Iterable[str], expected: Sequence[str]
) -> float | None:
    """Share of citations that were retrieved AND come from an expected source."""
    if not cited or not expected:
        return None
    retrieved_ids = set(retrieved_ids)
    good = [c for c in cited if c in retrieved_ids and _source_of(c) in set(expected)]
    return len(good) / len(cited)


def citation_completeness(cited: Sequence[str], expected: Sequence[str]) -> float | None:
    """Share of expected sources that the answer cites."""
    if not expected:
        return None
    cited_sources = {_source_of(c) for c in cited}
    return len(cited_sources & set(expected)) / len(set(expected))


def answer_points(answer: str, points: Sequence[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Keyword proxy for answer correctness: a point is covered if any keyword appears."""
    text = answer.lower()
    covered, missed = [], []
    for p in points:
        (covered if any(k.lower() in text for k in p["keywords"]) else missed).append(p["point"])
    return covered, missed


# -- aggregation ---------------------------------------------------------------------------------


def mean_of(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(statistics.fmean(vals), 4) if vals else None


def percentile(values: Sequence[float], q: float) -> float | None:
    vals = sorted(values)
    if not vals:
        return None
    if len(vals) == 1:
        return round(vals[0], 3)
    pos = (len(vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return round(vals[lo] + (vals[hi] - vals[lo]) * (pos - lo), 3)


def describe(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": round(min(values), 4),
        "median": round(statistics.median(values), 4),
        "max": round(max(values), 4),
    }


def item_metrics(item: dict[str, Any], result: dict[str, Any], n_relevant: int) -> dict[str, Any]:
    """All per-item metrics from one dataset row and one `AnswerResult.to_dict()`."""
    expected = item["expected_sources"]
    ranked = sorted(result["retrieved"], key=lambda c: c["rank"])
    sources = [c["source"] for c in ranked]
    ids = [c["chunk_id"] for c in ranked]
    cited = result["citations"]
    covered, missed = answer_points(result["answer"], item["expected_answer_points"])
    n_points = len(item["expected_answer_points"])
    return {
        "hit@k": hit_at_k(sources, expected),
        "recall@k": recall_at_k(sources, expected),
        "precision@k": precision_at_k(sources, expected),
        "mrr": mrr(sources, expected),
        "ndcg@k": ndcg_at_k(sources, expected, n_relevant),
        "citation_correctness": citation_correctness(cited, ids, expected),
        "citation_completeness": citation_completeness(cited, expected)
        if result["status"] in {"answered", "partial"}
        else None,
        "status_correct": result["status"] == item["expected_status"],
        "answer_point_coverage": round(len(covered) / n_points, 4) if n_points else None,
        "points_covered": covered,
        "points_missed": missed,
    }


METRIC_KEYS = (
    "hit@k",
    "recall@k",
    "precision@k",
    "mrr",
    "ndcg@k",
    "citation_correctness",
    "citation_completeness",
    "answer_point_coverage",
)


def summarise(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Means over the items where each metric applies, plus counts, latency and tokens."""
    ok = [i for i in items if i["status"] != "error"]
    summary: dict[str, Any] = {
        "items": len(items),
        "errors": len(items) - len(ok),
        "status_accuracy": mean_of(float(i["metrics"]["status_correct"]) for i in ok),
        "metrics": {
            key: {
                "mean": mean_of(i["metrics"][key] for i in ok),
                "n": sum(i["metrics"][key] is not None for i in ok),
            }
            for key in METRIC_KEYS
        },
        "abstention": _abstention(ok),
        "guardrail": {
            "first_try_pass_rate": mean_of(
                float(i["guardrail"]["attempts"] <= 1 and not i["guardrail"]["fallback"])
                for i in ok
                if i["guardrail"]["attempts"]
            ),
            "fallbacks": sum(i["guardrail"]["fallback"] for i in ok),
            "mean_attempts": mean_of(i["guardrail"]["attempts"] for i in ok),
        },
        "injection": {
            "items": sum(i["category"] == "injection" for i in items),
            "flagged": sum(
                bool(i["injection_flags"]) for i in items if i["category"] == "injection"
            ),
            "false_flags": sum(
                bool(i["injection_flags"]) for i in items if i["category"] != "injection"
            ),
        },
        "latency_s": {
            "median": percentile([i["latency_s"] for i in items], 0.5),
            "p95": percentile([i["latency_s"] for i in items], 0.95),
            "max": round(max((i["latency_s"] for i in items), default=0.0), 3),
        },
        "tokens": _tokens(items),
    }
    return summary


def _abstention(items: list[dict[str, Any]]) -> dict[str, Any]:
    should = [i for i in items if i["expected_status"] == "not_found"]
    did = [i for i in items if i["status"] == "not_found"]
    return {
        "should_abstain": len(should),
        "abstained_correctly": sum(i["status"] == "not_found" for i in should),
        "abstained_wrongly": sum(i["expected_status"] != "not_found" for i in did),
    }


def _tokens(items: list[dict[str, Any]]) -> dict[str, Any]:
    usage = [i["token_usage"] for i in items if i.get("token_usage")]
    if not usage:
        return {"reported": False}
    return {
        "reported": True,
        "total": sum(u["total"] for u in usage),
        "prompt": sum(u["prompt"] for u in usage),
        "completion": sum(u["completion"] for u in usage),
        "requests": sum(u["requests"] for u in usage),
        "mean_per_question": mean_of(u["total"] for u in usage),
    }


# -- running the dataset -------------------------------------------------------------------------


def run_dataset(
    items: list[dict[str, Any]],
    *,
    kb: Any,
    settings: Any,
    llm_factory: Any,
    tracer: Any,
    session_id: str,
    progress: Any = None,
) -> list[dict[str, Any]]:
    """Ask every question through the real crew and attach its metrics."""
    from app.crew import answer_question

    chunks_per_source = {s["source"]: s["chunks"] for s in kb.list_sources()}
    rows = []
    for item in items:
        result = answer_question(
            item["question"],
            kb=kb,
            settings=settings,
            llm_factory=llm_factory,
            tracer=tracer,
            session_id=session_id,
        ).to_dict()
        n_relevant = sum(chunks_per_source.get(s, 0) for s in item["expected_sources"])
        metrics = item_metrics(item, result, n_relevant)
        row = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "expected_status": item["expected_status"],
            "expected_sources": item["expected_sources"],
            "status": result["status"],
            "answer": result["answer"],
            "citations": result["citations"],
            "retrieved": result["retrieved"],
            "missing_evidence": result["missing_evidence"],
            "points_covered": metrics.pop("points_covered"),
            "points_missed": metrics.pop("points_missed"),
            "metrics": metrics,
            "latency_s": result["latency_s"],
            "token_usage": result["token_usage"],
            "guardrail": result["guardrail"],
            "injection_flags": result["injection_flags"],
            "model": result["model"],
            "trace_id": result["trace_id"],
            "error": result["error"],
        }
        rows.append(row)
        if progress:
            progress(row)
    return rows


def score_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Retrieval scores of relevant vs irrelevant chunks, to justify MIN_RELEVANCE (analysis A3)."""
    relevant, irrelevant, unanswerable_top = [], [], []
    for row in rows:
        expected = set(row["expected_sources"])
        for chunk in row["retrieved"]:
            (relevant if chunk["source"] in expected else irrelevant).append(chunk["score"])
        if row["expected_status"] == "not_found" and row["retrieved"]:
            unanswerable_top.append(max(c["score"] for c in row["retrieved"]))
    return {
        "relevant_chunks": describe(relevant),
        "irrelevant_chunks": describe(irrelevant),
        "top_score_on_unanswerable": describe(unanswerable_top),
    }
