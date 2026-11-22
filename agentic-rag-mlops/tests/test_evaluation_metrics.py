"""Metric functions used by scripts/run_evaluation.py, and the evaluation dataset itself."""

from __future__ import annotations

import math

import pytest

from app import evaluation as ev
from app.config import PROJECT_ROOT
from tests.conftest import SAMPLE_DIR

E = ["a.md"]


def test_hit_recall_precision_mrr():
    sources = ["x.md", "a.md", "a.md", "y.md"]
    assert ev.hit_at_k(sources, E) == 1.0
    assert ev.recall_at_k(sources, ["a.md", "b.md"]) == 0.5
    assert ev.precision_at_k(sources, E) == 0.5
    assert ev.mrr(sources, E) == 0.5
    assert ev.hit_at_k(["x.md"], E) == 0.0 and ev.mrr(["x.md"], E) == 0.0


def test_metrics_do_not_apply_without_expected_sources():
    for fn in (ev.hit_at_k, ev.recall_at_k, ev.precision_at_k, ev.mrr):
        assert fn(["a.md"], []) is None
    assert ev.ndcg_at_k(["a.md"], [], 3) is None


def test_ndcg_perfect_and_imperfect_ranking():
    assert ev.ndcg_at_k(["a.md", "a.md", "x.md"], E, n_relevant=2) == pytest.approx(1.0)
    worse = ev.ndcg_at_k(["x.md", "a.md", "a.md"], E, n_relevant=2)
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert worse == pytest.approx(expected)


def test_citation_correctness_and_completeness():
    retrieved = ["a.md#0", "b.md#1", "c.md#2"]
    assert ev.citation_correctness(["a.md#0", "b.md#1"], retrieved, ["a.md", "b.md"]) == 1.0
    assert ev.citation_correctness(["a.md#0", "c.md#2"], retrieved, ["a.md"]) == 0.5
    assert ev.citation_correctness(["ghost.md#0"], retrieved, ["ghost.md"]) == 0.0
    assert ev.citation_correctness([], retrieved, ["a.md"]) is None
    assert ev.citation_completeness(["a.md#0", "a.md#3"], ["a.md", "b.md"]) == 0.5


def test_answer_points_keyword_proxy():
    points = [{"point": "LeNet", "keywords": ["lenet"]}, {"point": "GPU", "keywords": ["gpu"]}]
    covered, missed = ev.answer_points("It is a LeNet-5 network.", points)
    assert covered == ["LeNet"] and missed == ["GPU"]


def test_percentiles_and_summary_skip_unmeasured_values():
    assert ev.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert ev.mean_of([None, 1, 0]) == 0.5 and ev.mean_of([None]) is None


def test_dataset_meets_the_assignment_minimum_and_points_to_real_files():
    items = ev.load_dataset(
        PROJECT_ROOT / "evaluation" / "questions.jsonl",
        PROJECT_ROOT / "evaluation" / "expected_answers.jsonl",
    )
    categories = [i["category"] for i in items]
    assert len(items) >= 5
    assert categories.count("answerable") >= 3
    assert "unanswerable" in categories and "multi_chunk" in categories
    sample_files = {p.name for p in SAMPLE_DIR.iterdir()}
    for item in items:
        assert set(item["expected_sources"]) <= sample_files, item["id"]
        assert item["expected_answer_points"], item["id"]
        assert item["expected_status"] in {"answered", "partial", "not_found"}
    multi = next(i for i in items if i["category"] == "multi_chunk")
    assert len(multi["expected_sources"]) >= 2


def test_run_dataset_end_to_end_with_the_stub_llm(kb, settings):
    """The evaluation loop works offline; real numbers come only from the live script."""
    from app.ingestion import ingest_files
    from app.observability import NoopTracer
    from tests.test_crew_smoke import cite_first, scripted_llm

    ingest_files([(p.name, p.read_bytes()) for p in sorted(SAMPLE_DIR.iterdir())], kb, settings)
    items = ev.load_dataset(
        PROJECT_ROOT / "evaluation" / "questions.jsonl",
        PROJECT_ROOT / "evaluation" / "expected_answers.jsonl",
    )[:3]
    rows = ev.run_dataset(
        items,
        kb=kb,
        settings=settings,
        llm_factory=lambda: scripted_llm(cite_first),
        tracer=NoopTracer(),
        session_id="test",
    )
    assert [r["id"] for r in rows] == ["q1", "q2", "q3"]
    assert all(r["metrics"]["hit@k"] is not None for r in rows)
    summary = ev.summarise(rows)
    assert summary["items"] == 3 and summary["errors"] == 0
    assert summary["metrics"]["hit@k"]["n"] == 3
    assert "relevant_chunks" in ev.score_distribution(rows)
