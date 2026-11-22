"""Run the evaluation dataset through the real workflow and save measured metrics (FR-022).

    uv run python scripts/run_evaluation.py

Needs OPENAI_API_KEY in .env. Uses a fresh temporary knowledge base with the sample files, so the
app's own knowledge base is untouched. With Langfuse configured, every question becomes a trace,
and all traces of one run share a Langfuse session named after the run id.

Output: evaluation/results/latest.json, plus a copy named after the run id.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import evaluation as ev  # noqa: E402
from app.config import APP_VERSION, get_settings  # noqa: E402
from app.ingestion import ingest_files  # noqa: E402
from app.services import build_services  # noqa: E402

SAMPLES = ROOT / "data" / "sample-knowledge"
RESULTS = ROOT / "evaluation" / "results"


def print_row(row: dict) -> None:
    m = row["metrics"]
    print(
        f"  {row['id']:<4} {row['category']:<13} status={row['status']:<10} "
        f"expected={row['expected_status']:<10} hit={m['hit@k']} cites={len(row['citations'])} "
        f"{row['latency_s']:.1f}s trace={row['trace_id']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation dataset.")
    parser.add_argument("--only", nargs="*", help="question ids to run (default: all)")
    parser.add_argument("--model", help="override OPENAI_MODEL for this run")
    parser.add_argument("--top-k", type=int, help="override TOP_K for this run")
    parser.add_argument(
        "--experiment",
        metavar="LABEL",
        help="save to evaluation/results/experiments/ and leave latest.json (the baseline) alone",
    )
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="rag-eval-"))
    overrides: dict = {"chroma_dir": tmp}
    if args.model:
        overrides["openai_model"] = args.model
    if args.top_k:
        overrides["top_k"] = args.top_k
    settings = get_settings().model_copy(update=overrides)
    svc = build_services(settings)
    if svc.llm_factory is None:
        print("OPENAI_API_KEY is not set in .env; the evaluation needs the real model.")
        return 2

    prefix = f"exp-{args.experiment}" if args.experiment else "eval"
    run_id = f"{prefix}-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    print(f"Run {run_id} | {svc.tracer.description}")

    files = [(p.name, p.read_bytes()) for p in sorted(SAMPLES.iterdir()) if p.is_file()]
    started = time.perf_counter()
    ingest = ingest_files(files, svc.kb, settings, svc.tracer)
    ingest_latency = time.perf_counter() - started
    indexed = sum(r.status == "indexed" for r in ingest)
    print(
        f"Indexed {indexed}/{len(files)} files ({svc.kb.count()} chunks) in {ingest_latency:.1f}s"
    )

    items = ev.load_dataset(
        ROOT / "evaluation" / "questions.jsonl", ROOT / "evaluation" / "expected_answers.jsonl"
    )
    if args.only:
        items = [i for i in items if i["id"] in set(args.only)]

    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    rows = ev.run_dataset(
        items,
        kb=svc.kb,
        settings=settings,
        llm_factory=svc.llm_factory,
        tracer=svc.tracer,
        session_id=run_id,
        progress=print_row,
    )
    svc.tracer.flush()

    report = {
        "run_id": run_id,
        "experiment": args.experiment,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "app_version": APP_VERSION,
        "environment": {"python": platform.python_version(), "platform": platform.system()},
        "config": {
            "model": settings.openai_model,
            "embedding_model": settings.embedding_model,
            "top_k": settings.top_k,
            "min_relevance": settings.min_relevance,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
        "tracing_enabled": svc.tracer.enabled,
        "langfuse_session_id": run_id if svc.tracer.enabled else None,
        "ingestion": {
            "files": len(files),
            "indexed": indexed,
            "rejected": len(files) - indexed,
            "chunks": svc.kb.count(),
            "latency_s": round(ingest_latency, 3),
        },
        "summary": ev.summarise(rows),
        "score_distribution": ev.score_distribution(rows),
        "items": rows,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.experiment:
        out = RESULTS / "experiments" / f"{run_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "latest.json").write_text(text, encoding="utf-8")
        out = RESULTS / f"{run_id}.json"
        out.write_text(text, encoding="utf-8")

    s = report["summary"]
    print(
        f"\nStatus accuracy {s['status_accuracy']} | Hit@k {s['metrics']['hit@k']['mean']} | "
        f"MRR {s['metrics']['mrr']['mean']} | citation correctness "
        f"{s['metrics']['citation_correctness']['mean']} | median latency "
        f"{s['latency_s']['median']} s"
    )
    print(f"Wrote {out}" + ("" if args.experiment else " and latest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
