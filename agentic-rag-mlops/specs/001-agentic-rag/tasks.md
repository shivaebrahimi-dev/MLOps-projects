---
description: "Implementation tasks for the Agentic RAG Knowledge Assistant"
---

# Tasks: Agentic RAG Knowledge Assistant

**Input**: Design documents from `specs/001-agentic-rag/`

**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md),
[contracts/](contracts/)

**Tests**: Required by the spec (FR-020) and the constitution (Principle V). Test tasks come before the matching
implementation task in each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no dependency on unfinished tasks)
- **[Story]**: the user story (US1–US5) the task belongs to
- Paths are relative to the project root `agentic-rag-mlops/`

## Phase 1: Setup (shared infrastructure)

- [x] T001 Create the project folder, `pyproject.toml`, `uv.lock`, `requirements.txt`, `requirements-dev.txt`, `.gitignore`
- [x] T002 Install Spec Kit (`.specify/`, `.claude/skills/speckit-*`) and the official CrewAI and Langfuse skills (`skills-lock.json`)
- [x] T003 [P] Write `.env.example` with variable names only (FR-024)
- [x] T004 [P] Copy the constitution to `specs/constitution.md`

## Phase 2: Foundational (blocks every user story)

- [x] T005 Implement `Settings` in `app/config.py`: `.env` loading, empty value means default, no key ever in `repr` (Principle VI)
- [x] T006 [P] Implement the exception types in `app/errors.py` (`UploadRejected` with a reason)
- [x] T007 [P] Implement the models in `app/models.py` from [data-model.md](data-model.md) (`Chunk`, `RetrievedChunk`, `IngestResult`, `AnswerDraft`, `AnswerResult`, `SAFE_ANSWER`)
- [x] T008 [P] Implement the `Embedder` protocol, `OpenAIEmbedder`, and `HashEmbedder` in `app/embeddings.py`
- [x] T009 Implement `KnowledgeBase` in `app/knowledge_base.py`: upsert, delete by source, list sources, clear, query; cosine space; Chroma telemetry off; if the store folder disappears, recreate the collection and report an empty knowledge base instead of crashing (analysis A6)
- [x] T010 Implement `NoopTracer` and the `Tracer` interface in `app/observability.py`, so other modules can depend on it before Langfuse is wired
- [x] T011 [P] Add `tests/conftest.py`: env isolation (no real keys, tracing off), a temporary knowledge base with `HashEmbedder`; copy `tests/stub_llm.py` from the CrewAI skill

**Checkpoint**: settings, storage, and test fixtures exist.

## Phase 3: User Story 1 — Upload and index knowledge files (P1) 🎯 MVP part 1

**Goal**: Valid files are chunked, embedded, and stored with metadata; invalid files are rejected with a reason.

**Independent Test**: `pytest tests/test_validation.py tests/test_chunking.py`, then manual scenarios M1–M3 in [quickstart.md](quickstart.md).

- [x] T012 [P] [US1] Tests in `tests/test_validation.py`: unsupported type, empty file, whitespace-only file, PDF without text, oversized file, valid txt/md/pdf (FR-001, FR-002)
- [x] T013 [P] [US1] Tests in `tests/test_chunking.py`: chunk size and overlap, ids `file#n`, PDF page numbers, source metadata preserved, re-upload replaces (FR-003, FR-005), two files with the same name in one upload: the later one wins and its message says so (analysis A1)
- [x] T014 [US1] Implement `validate_upload`, `extract_text`, `chunk_text`, `ingest_files` in `app/ingestion.py`, processing in memory only (FR-007), including the same-name-in-one-upload rule (analysis A1)
- [x] T015 [US1] Wrap ingestion in an `ingest` span through the tracer interface

**Checkpoint**: files can be indexed from Python.

## Phase 4: User Story 2 — Ask a question and get a cited answer (P1) 🎯 MVP part 2

**Goal**: A two-agent sequential crew answers from retrieved evidence with `[file#chunk]` citations.

**Independent Test**: `pytest tests/test_retrieval.py tests/test_crew_smoke.py`; manual M4–M5.

- [x] T016 [P] [US2] Tests in `tests/test_retrieval.py`: a known question returns the expected source in the top k; scores ordered; threshold sets `is_evidence`; evidence formatting is delimited
- [x] T017 [US2] Implement `retrieve()` and `format_evidence()` in `app/retrieval.py`, with a `retriever` observation (FR-009, FR-013)
- [x] T018 [US2] Implement `KnowledgeSearchTool` and its per-run `RetrievalLog` in `app/crew.py`
- [x] T019 [US2] Implement the Retriever and Answer/Verifier agents, tasks, and prompts (evidence only, cite every claim, separate evidence from uncertainty) and `build_crew()` in `app/crew.py` (FR-009, FR-010)
- [x] T020 [US2] Implement `answer_question()` in `app/crew.py`: fresh LLMs per run, `AnswerResult` assembly, token usage, latency; reject empty questions and cut questions over `MAX_QUESTION_CHARS` here (not only in the UI), with a test (analysis A2)
- [x] T021 [US2] Smoke test in `tests/test_crew_smoke.py`: ingest the sample docs, run the crew with `StubLLM`, assert the answer has citations that resolve to retrieved chunks

**Checkpoint**: the MVP works from Python.

## Phase 5: User Story 3 — Safe answer when the evidence is missing (P2)

**Goal**: A hallucination guardrail; Not found and Partial statuses; injection flags.

**Independent Test**: `pytest tests/test_guardrails.py`; manual M6–M7.

- [x] T022 [P] [US3] Tests in `tests/test_guardrails.py`: unknown citation rejected; answer without citations rejected; `not_found` forced when nothing reaches the threshold; partial without missing evidence rejected; injection text detected; a valid draft passes
- [x] T023 [US3] Implement `SAFE_ANSWER`, `detect_injection`, `check_answer`, `parse_draft`, and the guardrail factories in `app/guardrails.py` (FR-011, FR-012, FR-014)
- [x] T024 [US3] Attach the guardrails to both tasks; catch the error after the last retry and fall back to the safe answer in `app/crew.py`
- [x] T025 [US3] Add smoke tests for the unanswerable path (stub LLM tries to cite a chunk that was not retrieved, so the fallback is used) and the injection path to `tests/test_crew_smoke.py`

## Phase 6: User Story 4 — Trace every question (P2)

**Goal**: One Langfuse trace per question with nested agent, task, LLM, retrieval, and guardrail observations.

**Independent Test**: `pytest tests/test_observability.py`; manual M8–M10 (needs keys).

- [x] T026 [P] [US4] Tests in `tests/test_observability.py`: tracing off without keys; `NoopTracer` works with the crew; a fake Langfuse client receives the root observation with session id, request id, and no secrets; errors recorded with level ERROR
- [x] T027 [US4] Implement `LangfuseTracer` and `init_tracing()` in `app/observability.py`: Langfuse client, `auth_check`, CrewAI and OpenAI instrumentors (once), `question_trace` with `propagate_attributes`, `retriever` and `guardrail` observations, trace id and URL, flush (FR-015 to FR-019)
- [x] T028 [US4] Implement `app/services.py` wiring (settings → embedder, knowledge base, tracer, LLM factory; telemetry env vars)
- [x] T029 [US4] Add `tests/test_live.py` (marked `live`): one real question with OpenAI and Langfuse, assert a trace id is returned

## Phase 7: UI for US1–US4

**Goal**: The Streamlit two-tab UI in [contracts/ui.md](contracts/ui.md).

- [x] T030 [US1] Knowledge tab in `app/ui.py`: uploader, Index button, results table, summary, knowledge base panel, clear with confirmation
- [x] T031 [US2] Ask tab in `app/ui.py`: empty-knowledge notice, question input, status box, badge, answer, missing evidence, source expanders, run details with trace link
- [x] T032 [US4] Header tracing badge and per-session UUID `session_id` in `app/ui.py`
- [ ] T033 Run manual scenarios M1–M10 and note the results for the README

## Phase 8: Sample knowledge and evaluation dataset (supports US5)

- [x] T034 [P] Write the sample knowledge in `data/sample-knowledge/`, from the author's own HW1–HW3 work: at least one `.md`, one `.txt`, one `.pdf` (generated by `scripts/make_sample_pdf.py`), and one document with a planted instruction
- [x] T035 [P] Write `evaluation/questions.jsonl` and `evaluation/expected_answers.jsonl` from [contracts/evaluation-dataset.md](contracts/evaluation-dataset.md): at least 3 answerable, 1 unanswerable, 1 multi-document, plus partial and injection items (FR-021)
- [x] T036 [P] Tests in `tests/test_evaluation_metrics.py` for the metric functions (hit, recall, precision, MRR, nDCG, citation metrics)
- [x] T037 Implement `scripts/run_evaluation.py`: fresh temporary knowledge base, ingest the samples, run every question, compute metrics, write `evaluation/results/latest.json` (FR-022); record the retrieval score distribution of relevant and irrelevant chunks so the report can justify `MIN_RELEVANCE` (analysis A3), and the median end-to-end latency for SC-007 (analysis A5)
- [x] T038 Implement `scripts/export_langfuse_traces.py`: read the traces by id from the Langfuse API, write `reports/langfuse_traces_summary.json` (latency, tokens, cost, model names, observation counts by type, levels), and write nothing it did not receive. Report explicitly whether token usage and model name arrived on the generation observations (analysis A4)

## Phase 9: User Story 5 — Evaluate the system on request (P3)

**Goal**: The `rag-evaluation-report` skill writes an honest report when the user asks for it.

**Independent Test**: the user invokes `/rag-evaluation-report`; every number in the report can be traced to a file.

- [x] T039 [US5] Write `.agents/skills/rag-evaluation-report/SKILL.md` (frontmatter `name`, `description`, `user-invocable: true`, `disable-model-invocation: true`): inputs, the metric catalogue based on Homework 3, measured/unavailable rules, report template, privacy rules (FR-023)
- [x] T040 [US5] Mirror the skill to `.claude/skills/rag-evaluation-report/SKILL.md` and add a test that the frontmatter is the same in both and has the required values
- [ ] T041 [US5] The user runs the evaluation and invokes the skill; commit `reports/rag-evaluation-report.md`

## Phase 10: Polish and submission

- [ ] T042 Write `README.md`: setup (uv/pip, OpenAI, Langfuse Cloud keys, confirming traces arrive), running the UI, tests, manual procedure, architecture, skills, evaluation, security notes
- [ ] T043 Save the `pytest` output to `reports/test-results.txt`; `ruff check .` clean
- [ ] T044 Add the Langfuse trace screenshot (or trace id) to `reports/` and link it from the README
- [ ] T045 Add the project to the root `README.md`; scan for secrets and personal data; build the ZIP with `git archive`

## Dependencies and execution order

- Setup (Phase 1) → Foundational (Phase 2) → US1 → US2 → US3 and US4 (either order) → UI → evaluation data → US5
  → Polish.
- US2 needs US1 (it needs indexed chunks). US3 extends the US2 crew. US4 instruments US1–US3 through the tracer
  interface defined in T010, so the earlier stories do not wait for Langfuse.
- US5 needs the evaluation results (Phase 8) and the test results (T043, which can be run early and refreshed).

## Parallel opportunities

- Phase 2: T006, T007, T008, T011 touch different files.
- US1: T012 and T013 (tests) in parallel, then T014.
- Phase 8: T034, T035, and T036 in parallel.

## Implementation strategy

1. **MVP** = Phases 1–4: upload files and get cited answers from Python, with the smoke test passing.
2. Add safety (US3), then observability (US4), then the UI. Each phase ends with `pytest` and `ruff` passing and a
   commit.
3. Evaluation data, the evaluation scripts, and the skill come last, because they measure the finished system.

## Requirement coverage

| Requirement | Tasks |
|---|---|
| FR-001–FR-007 | T012–T015, T030 |
| FR-008–FR-013 | T016–T021, T031 |
| FR-011, FR-012, FR-014 | T022–T025 |
| FR-015–FR-019 | T026–T029, T032 |
| FR-020 | T011–T013, T016, T021, T022, T025, T026, T036 |
| FR-021–FR-022 | T034–T038 |
| FR-023 | T039–T041 |
| FR-024 | T003, T005 |
