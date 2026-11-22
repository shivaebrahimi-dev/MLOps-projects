# Implementation Plan: Agentic RAG Knowledge Assistant

**Branch**: `add-agentic-rag-assistant` | **Date**: 2026-11-22 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-agentic-rag/spec.md` (iteration 2, clarified)

## Summary

A single-user Streamlit app with two tabs.

- **Knowledge tab**: validates `.txt`, `.md`, and `.pdf` uploads in memory, chunks them with source, chunk, and page
  metadata, embeds the chunks with OpenAI, and stores them in a local ChromaDB collection.
- **Ask tab**: runs a sequential CrewAI crew.
  1. A **Retriever** agent calls a `search_knowledge_base` tool that returns scored, delimited evidence.
  2. An **Answer/Verifier** agent writes a structured `AnswerDraft` (status, answer, citations, missing evidence)
     from that evidence only.
- **Guardrail**: a deterministic function guardrail rejects citations that were not retrieved and answers that cite
  nothing. When nothing reaches the relevance threshold, it returns the safe "not found" answer.
- **Tracing**: each question is one Langfuse trace. A root `agent` observation holds the OpenInference CrewAI and
  OpenAI spans (agents, tasks, LLM calls, tokens, model) plus our own `retriever` and `guardrail` observations.
- **Tests**: run offline with a stub LLM, a hash embedder, and a no-op tracer.
- **Evaluation**: a script runs the dataset through the real crew and saves the measured metrics. The user-invoked
  `rag-evaluation-report` skill turns those files into an honest report.

See [research.md](research.md) for every decision and the API facts checked against the installed packages.

## Technical Context

**Language/Version**: Python 3.12

**Primary Dependencies**:
- crewai 1.15.44
- langfuse 5.2.0
- openinference-instrumentation-crewai 1.1.30 and openinference-instrumentation-openai 0.1.77
- openai 2.54
- chromadb 1.5.9
- pypdf 7.1
- streamlit 1.67
- pydantic-settings 2.16

Dev dependencies: pytest 9, ruff 0.17, and reportlab (to generate the sample PDF).

**Storage**: ChromaDB persistent collection in `CHROMA_DIR` (default `.chroma/`, gitignored). Uploaded files are not
stored (FR-007).

**Testing**: pytest, offline. Uses `tests/stub_llm.py` (from the CrewAI `test-crewai-project` skill), `HashEmbedder`,
a temporary Chroma directory, and `NoopTracer`/a fake Langfuse client. Live tests are marked `live` and skipped unless
`RUN_LIVE_TESTS=1`.

**Target Platform**: Local workstation (Windows, macOS, Linux) with a browser. Needs internet access for OpenAI and
Langfuse Cloud.

**Project Type**: Single Python project: a Streamlit app plus scripts.

**Performance Goals**: A typical question is answered in under 30 s (SC-007). Indexing the sample set takes under
30 s.

**Constraints**:
- Upload limit: 10 MB per file.
- The tests run with no network.
- No secrets in git, logs, or traces.
- Temperature 0.

**Scale/Scope**: Tens of documents, about a thousand chunks at most, one user.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | How the design satisfies it | Status |
|---|---|---|
| I. Grounded answers or no answer | Evidence-only prompts; `AnswerDraft` with `status`; a function guardrail that checks citations against retrieved ids; threshold-based `not_found`; a safe fallback after the retries are used up | PASS |
| II. Specification before code | Spec v1 and its clarified v2 are committed before any code; `tasks.md` maps tasks to FR ids; `CHANGELOG.md` records iterations | PASS |
| III. Separation of concerns | `app/ui.py` is the only Streamlit importer. Ingestion, knowledge base, retrieval, crew, guardrails, observability, and config are separate modules. Embedder, LLM, and tracer are injected | PASS |
| IV. Observable by default, private by design | One trace per question, with session/request UUIDs; retriever and guardrail observations; OpenInference agent, task, and LLM spans; ERROR level on failure; `NoopTracer`; CrewAI and Chroma telemetry off; snippets only in the retrieval span | PASS |
| V. Deterministic, offline tests | Stub LLM, hash embedder, temporary store, no-op tracer; live tests opt-in | PASS |
| VI. Secrets stay out of source control | `.env` gitignored; `.env.example` holds names only; settings never log keys; ZIP built with `git archive` | PASS |
| VII. Honest evaluation and simplicity | Metrics come from `evaluation/results/latest.json` and the exported trace summaries; the skill labels the rest unavailable. Two agents, one store, one UI | PASS |

Post-design re-check (after Phase 1): still **PASS**. Two design choices needed checking:
- Separate `knowledge_base.py` and `retrieval.py` modules keep storage and the evidence policy apart. This adds no
  service.
- The retriever-task guardrail enforces that the tool was actually used, which supports Principle I.

## Project Structure

### Documentation (this feature)

```text
specs/
├── constitution.md               # copy of .specify/memory/constitution.md (assignment requires it here)
└── 001-agentic-rag/
    ├── spec.md                   # requirements (iteration 2)
    ├── CHANGELOG.md              # spec iteration evidence
    ├── plan.md                   # this file
    ├── research.md               # Phase 0
    ├── data-model.md             # Phase 1
    ├── quickstart.md             # Phase 1: validation guide
    ├── contracts/
    │   ├── ui.md                 # what each tab shows and does
    │   ├── answer.schema.json    # AnswerDraft / AnswerResult shape
    │   ├── evaluation-dataset.md # JSONL formats for the dataset and results
    │   └── trace.md              # expected Langfuse trace shape
    ├── checklists/requirements.md
    └── tasks.md                  # /speckit-tasks
```

### Source Code (project root `agentic-rag-mlops/`)

```text
app/
├── __init__.py
├── config.py          # Settings (pydantic-settings, .env, empty value = default)
├── errors.py          # UploadRejected(reason) and subclasses
├── models.py          # Chunk, RetrievedChunk, IngestResult, AnswerDraft, AnswerResult
├── ingestion.py       # validate_upload, extract_text (txt/md/pdf), chunk_text, ingest_files
├── embeddings.py      # Embedder protocol, OpenAIEmbedder, HashEmbedder
├── knowledge_base.py  # KnowledgeBase over Chroma: upsert_document, delete_source, list_sources, clear, query
├── retrieval.py       # retrieve(question, k) -> list[RetrievedChunk]; threshold; injection screen; evidence formatting
├── guardrails.py      # SAFE_ANSWER, detect_injection, check_answer (citation rules), make_*_guardrail
├── crew.py            # build_crew(...), answer_question(...) -> AnswerResult; KnowledgeSearchTool
├── observability.py   # init_tracing(settings) -> Tracer (LangfuseTracer | NoopTracer); instrumentors
├── services.py        # wiring: get_services(settings) builds embedder, KB, tracer, LLM factory
└── ui.py              # Streamlit app: Knowledge and Ask tabs (only module importing streamlit)
data/sample-knowledge/ # original docs from the author's HW1-HW3 projects (.md, .txt, .pdf) + one injection test doc
evaluation/
├── questions.jsonl
├── expected_answers.jsonl
└── results/           # written by scripts/run_evaluation.py
scripts/
├── make_sample_pdf.py
├── run_evaluation.py
└── export_langfuse_traces.py
tests/
├── conftest.py        # tmp KB, hash embedder, stub LLM fixtures, env isolation
├── stub_llm.py        # from the CrewAI test-crewai-project skill
├── test_validation.py
├── test_chunking.py
├── test_retrieval.py
├── test_guardrails.py
├── test_observability.py
├── test_crew_smoke.py
├── test_evaluation_metrics.py
└── test_live.py       # @pytest.mark.live
.agents/skills/rag-evaluation-report/SKILL.md   # student skill (assignment location)
.claude/skills/rag-evaluation-report/SKILL.md   # same file, so Claude Code lists it as /rag-evaluation-report
reports/               # test-results.txt, langfuse_traces_summary.json, trace screenshot, rag-evaluation-report.md
```

**Structure Decision**: A single project. The flat `app/` package follows the assignment's suggested layout
(`ui.py`, `ingestion.py`, `retrieval.py`, `crew.py`, `observability.py`, `config.py`). It adds `knowledge_base.py`,
`guardrails.py`, `embeddings.py`, `models.py`, `errors.py`, and `services.py`, so that each has one responsibility
(Principle III).

## Key flows

**Ingest** (Knowledge tab → `ingestion.ingest_files`):
1. Check each upload: extension, size, then text (empty, whitespace-only, or a PDF with no text). A failed check
   gives `IngestResult(status="rejected", reason)`.
2. Split into chunks, then embed them in one batch.
3. `KnowledgeBase.upsert_document` deletes the old chunks for that source, then adds the new ones.
4. Return `IngestResult(status="indexed", chunks=n)`. All of this runs inside an `ingest` span when tracing is on.

**Ask** (Ask tab → `crew.answer_question`):
1. `tracer.question_trace(question, session_id)` opens the root observation and `propagate_attributes`.
2. `build_crew` creates fresh `LLM` objects and a `KnowledgeSearchTool`. The tool keeps a per-run
   `RetrievalLog` of every chunk it returns.
3. **Retriever task**: the agent calls the tool, which calls `retrieval.retrieve` inside a `retriever` observation.
   The task's guardrail requires that the log is not empty.
4. **Answer task**: the agent sees the evidence through task context and returns `AnswerDraft` JSON. The function
   guardrail parses `output.raw`, runs `check_answer(draft, log)`, records a `guardrail` observation, and either
   passes, sends feedback for a retry, or replaces the draft with the safe answer.
5. Build `AnswerResult`: status, answer, citations resolved to `RetrievedChunk`s, missing evidence, injection flags,
   token usage, latency, trace id, and trace URL. Set the root observation's output, then flush.
6. On any exception, record level=ERROR on the root observation and return
   `AnswerResult(status="error", message=...)`. The UI shows the message.

## Complexity Tracking

No constitution violations to justify.
