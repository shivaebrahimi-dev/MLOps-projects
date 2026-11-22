# Agentic RAG Knowledge Assistant Constitution

## Core Principles

### I. Grounded Answers or No Answer

Every answer MUST be supported by chunks retrieved from the user's uploaded knowledge.

- The answer agent MUST use only the retrieved context. It MUST NOT add facts from model memory.
- Every factual statement in an answer MUST cite at least one retrieved chunk as `[source#chunk]`.
- When retrieval finds no chunk above the relevance threshold, or the evidence does not cover the
  question, the system MUST reply with the safe abstention: "I could not find this in the uploaded
  knowledge." It MAY add what evidence is missing.
- A hallucination guardrail MUST check each answer in code before the user sees it. It MUST reject
  citations that do not match a retrieved chunk, and answers that are not abstentions but cite
  nothing.

Rationale: a knowledge assistant that invents facts is worse than one that says it does not know.

### II. Specification Before Code

Requirements, design, acceptance criteria, and tasks MUST be written in `specs/` with Spec Kit
before implementation code is written.

- Each implementation task MUST trace back to a requirement in `spec.md`.
- A change of scope MUST update the specification first, and the change MUST be recorded in the
  feature's `CHANGELOG.md`.

### III. Separation of Concerns

The UI, ingestion, retrieval, agent orchestration, guardrails, observability, and configuration
MUST be separate Python modules with one responsibility each.

- The Streamlit module MUST contain only interface logic and MUST NOT be imported by other modules.
- Service modules MUST NOT import Streamlit, so that they can be tested without a UI.
- The LLM, the embedding model, and the vector store MUST be reachable through small interfaces, so
  that tests can replace them with test doubles.

### IV. Observable by Default, Private by Design

Every question MUST produce one Langfuse trace when tracing is enabled.

- The trace MUST contain the CrewAI agent, task, and LLM spans, a retrieval span with chunk ids,
  sources, and scores, the guardrail result, latency, status, and token usage and model name when
  the provider reports them.
- Failed runs MUST be recorded with an error level and message.
- Traces MUST use random, non-identifying session and request ids. They MUST NOT contain API keys,
  user names, emails, or other personal data.
- When tracing is disabled or its keys are missing, the application MUST keep working with a no-op
  tracer.

### V. Deterministic, Offline Tests

The automated test suite MUST run without network access, API keys, or a Langfuse account.

- Tests MUST use a deterministic fake LLM and a deterministic fake embedder.
- The suite MUST cover file validation, chunking and metadata, retrieval of known evidence,
  abstention on unknown questions, citations on answerable questions, disabled or mocked tracing,
  and an end-to-end smoke test.
- Tests that call real services MUST be marked `live` and skipped by default.

### VI. Secrets Stay Out of Source Control

API keys and other secrets MUST come from environment variables or a local `.env` file.

- `.env` MUST be ignored by git. `.env.example` MUST list variable names only, with no values.
- Logs, traces, error messages, and reports MUST NOT print secrets.
- The submitted ZIP MUST be built from tracked files only.

### VII. Honest Evaluation and Simplicity

- The evaluation report MUST label every metric as measured (with the file it came from) or
  unavailable (with how to collect it). It MUST NOT invent values, including Langfuse data.
- The design MUST stay small: a sequential two-agent crew, one local vector store, and one UI.
  Extra components MUST be justified in `plan.md`.

## Technology and Data Constraints

- Language: Python 3.12, with dependencies locked by `uv.lock` and exported to `requirements.txt`.
- Agent framework: CrewAI. Observability: Langfuse Cloud through the Langfuse Python SDK and
  OpenInference instrumentation. UI: Streamlit. Vector store: ChromaDB, persisted locally.
- Model provider: OpenAI for chat and embeddings. Model names MUST be configurable.
- Supported uploads: `.txt`, `.md`, and `.pdf`. Empty files, files without extractable text, files
  above the size limit, and other types MUST be rejected with a clear message.
- Sample knowledge and evaluation data MUST be original or openly licensed, so that an evaluator can
  access all of it. The sample documents are written from the author's own earlier course projects.

## Development Workflow and Quality Gates

- Work proceeds in the phases listed in `tasks.md`. Each completed phase is committed.
- A phase is complete only when `pytest` passes and `ruff check .` reports no errors.
- The `rag-evaluation-report` skill MUST declare `user-invocable: true` and
  `disable-model-invocation: true`. It runs only when the user explicitly asks for an evaluation
  report, after implementation and testing.
- Before any push, staged files MUST be scanned for secrets and personal data.

## Governance

This constitution takes precedence over other project practices. `plan.md` MUST include a
Constitution Check that confirms each principle, and any justified exception MUST be listed in its
Complexity Tracking table.

Amendments MUST be made in this file, with a version change and a note in the feature
`CHANGELOG.md`. Versions follow semantic versioning: MAJOR for removed or redefined principles,
MINOR for new principles or sections, PATCH for wording fixes.

**Version**: 1.0.0 | **Ratified**: 2026-11-22 | **Last Amended**: 2026-11-22
