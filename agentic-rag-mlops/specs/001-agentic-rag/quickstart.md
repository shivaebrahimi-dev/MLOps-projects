# Quickstart Validation Guide

Use this guide to show that the feature works end to end. Setup details are in the project [README](../../README.md).

## Prerequisites

- Python 3.12 and [uv](https://docs.astral.sh/uv/) (or pip)
- An OpenAI API key
- A Langfuse Cloud project with a public key and a secret key

## 1. Install and configure

```bash
uv sync
cp .env.example .env
```

Fill in `OPENAI_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL` in `.env`.

## 2. Offline tests (no keys needed)

```bash
uv run pytest -q
```

Expected: every test passes, and `live` tests are reported as skipped.

## 3. Manual scenarios

Start the app with `uv run streamlit run app/ui.py`.

| # | Action | Expected | Covers |
|---|---|---|---|
| M1 | Knowledge tab: upload every file in `data/sample-knowledge/` and click Index | Each file Indexed with a chunk count; success message | US1-1 |
| M2 | Upload an empty `.txt` and any `.docx` | Both Rejected: "empty file", "unsupported file type: .docx" | US1-2 |
| M3 | Upload one sample file again | Its chunk count is unchanged in the knowledge base panel (replaced, not duplicated) | US1-3 |
| M4 | Ask tab: ask evaluation question q1 | Processing state visible, then Answered with `[source#n]` citations; sources expand to passages | US2-1, US2-2 |
| M5 | Ask the multi-document question | The answer cites two different files | US2-3 |
| M6 | Ask the unanswerable question | "I could not find this in the uploaded knowledge.", status Not found, no citations | US3-1 |
| M7 | Ask the injection question | The answer does not follow the planted instruction; Run details shows an injection flag | US3-3 |
| M8 | Copy the trace id from Run details; in Langfuse open Tracing and search for it | One `rag-question` trace with retriever, agent, task, LLM (model and tokens), and guardrail observations | US4-1 |
| M9 | Set `LANGFUSE_ENABLED=false`, restart, ask q1 | The answer still works; the header says "Langfuse: off" | US4-2 |
| M10 | Set an invalid `OPENAI_API_KEY`, restart, ask q1 | A clear error message in the UI; with tracing on, the trace has level ERROR | US4-3 |

## 4. Evaluation and report

```bash
uv run python scripts/run_evaluation.py
uv run python scripts/export_langfuse_traces.py
```

The first command writes `evaluation/results/latest.json`. The second writes `reports/langfuse_traces_summary.json`.

Then open Claude Code in this folder and type `/rag-evaluation-report`. Expected: `reports/rag-evaluation-report.md`
exists, and every number in it can be found in the two files above or in `reports/test-results.txt`.
