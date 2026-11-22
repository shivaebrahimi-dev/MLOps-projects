# Research: Agentic RAG Knowledge Assistant

Phase 0 of `/speckit-plan`. Each decision was checked against the **installed** packages (`uv.lock`, 2026-11-22):
crewai 1.15.44, langfuse 5.2.0, openinference-instrumentation-crewai 1.1.30, openinference-instrumentation-openai
0.1.77, chromadb 1.5.9, openai 2.54.0, pypdf 7.1.1, streamlit 1.67.0. CrewAI facts come from the official CrewAI
`check-crewai-api` skill; Langfuse facts come from the official Langfuse skill and its CrewAI integration page,
then from the installed SDK's signatures.

## R1. Agent framework usage (CrewAI 1.15)

- **Decision**: A plain `Crew(process=Process.sequential)` built in Python, with two `Agent`s and two `Task`s. No
  `@CrewBase` YAML project and no `crewai create` scaffold.
- **Rationale**: Two agents fit in one readable module. A scaffold would add a second project layout inside the repo.
  The `crewai create crew` wizard is interactive and writes a JSON project, which does not fit a Streamlit app.
- **Alternatives**: `@CrewBase` with `agents.yaml`/`tasks.yaml` (more files for no gain); a CrewAI Flow (not needed
  for a fixed two-step sequence).
- **Verified API facts**:
  - `Task.expected_output` is required.
  - Unknown keyword arguments are silently dropped, so tests assert that used fields exist in `Task.model_fields`.
  - Structured output uses `Task(output_pydantic=Model)`. The result is in `out.pydantic`, and the code must check
    it is not `None`.
  - Guardrails are functions `(TaskOutput) -> tuple[bool, Any]`, set with `guardrails=[...]` and
    `guardrail_max_retries` (default 3). After the last retry the task raises an error.
  - `crewai.tasks.hallucination_guardrail.HallucinationGuardrail` is a **no-op in open-source CrewAI**. The
    hallucination guardrail is therefore our own function guardrail.
  - Tools are `crewai.tools.BaseTool` subclasses, or the `@tool` decorator from `crewai.tools`.
  - `token_usage` is summed per `LLM` object over its lifetime. Each crew run gets **fresh** `LLM` objects so that
    token counts are per run.

## R2. Language model provider

- **Decision**: OpenAI through CrewAI's native provider, `LLM(model="openai/<OPENAI_MODEL>", temperature=0)`,
  default `gpt-4o-mini`.
- **Rationale**: The user chose OpenAI. Core CrewAI ships the OpenAI SDK, and LiteLLM is no longer a dependency. A
  temperature of 0 makes runs as repeatable as the provider allows.
- **Alternatives**: Anthropic (needs the `crewai[anthropic]` extra); local Ollama (free, but small models follow
  citation instructions less reliably).

## R3. Deterministic fake LLM for tests

- **Decision**: A `BaseLLM` subclass (`tests/fakes.py`) that returns scripted replies: a ReAct-format tool call for
  the retriever agent, then the final answer JSON. It accepts the full `call` signature (`messages, tools, callbacks,
  available_functions, from_task, from_agent, response_model, **kwargs`).
- **Rationale**: Runs the real CrewAI orchestration, tool execution, guardrails and parsing offline. The CrewAI
  `test-crewai-project` skill ships a reference stub (`references/stub_llm.py`), which we follow.
- **Alternatives**: Mocking `Crew.kickoff` (would not test orchestration); recorded HTTP cassettes (brittle, and they
  need a key to record).

## R4. Embeddings and vector store

- **Decision**: ChromaDB `PersistentClient` in `CHROMA_DIR` (default `.chroma/`), one collection with cosine
  distance. Chroma receives **our** embeddings, through an `Embedder` protocol:
  - `OpenAIEmbedder` (`text-embedding-3-small`) in the app
  - `HashEmbedder` (deterministic bag-of-words hashing) in tests
- **Rationale**: Local, file-based, nothing to run as a server, and persistent across restarts (FR-007). Passing
  embeddings ourselves keeps tests offline.
- **Relevance score**: `score = 1 - cosine_distance`. A chunk counts as evidence only if `score >= MIN_RELEVANCE`
  (default 0.30, set from the evaluation run).
- **Privacy**: Chroma is created with `Settings(anonymized_telemetry=False)`.
- **Alternatives**: FAISS (no metadata filtering or persistence out of the box); CrewAI `knowledge_sources` (hides
  retrieval inside CrewAI, so we could not trace chunk ids and scores ourselves or control the threshold).

## R5. Parsing and chunking

- **Decision**:
  - `.txt` and `.md` are decoded as UTF-8, falling back to Latin-1.
  - `.pdf` uses `pypdf`, page by page; a PDF whose pages yield no text is rejected.
  - The chunker is recursive: split on blank lines, then lines, then sentences, then characters. Target size is
    `CHUNK_SIZE` (800 characters) with `CHUNK_OVERLAP` (120 characters). PDF chunks never cross pages, so each one
    keeps its page number.
  - Chunk ids are `<file name>#<n>`, starting at 0.
  - Re-uploading a file name deletes that file's chunks first (`where={"source": name}`).
- **Rationale**: No heavy dependency (LangChain is not needed for one splitter), and the behaviour is easy to test.
- **Alternatives**: Token-based splitting (adds a tokenizer dependency for little gain at this scale).

## R6. Hallucination guardrail and answer statuses

- **Decision**: The answer task uses `output_pydantic=AnswerDraft`, with fields `status`
  (`answered|partial|not_found`), `answer`, `citations`, and `missing_evidence`. A function guardrail checks the
  draft against the chunk ids the retriever actually returned:
  1. Every id in `citations`, and every inline `[file#n]` in `answer`, must be a retrieved id.
  2. `answered` and `partial` drafts need at least one citation, and `partial` needs `missing_evidence`.
  3. `not_found` must carry the exact safe message and no citations.
  4. If no retrieved chunk reached `MIN_RELEVANCE`, the guardrail does not ask the model. It returns the safe
     `not_found` answer.

  A second guardrail on the retriever task requires that the search tool was actually called. After
  `guardrail_max_retries` (2) failed attempts, the app catches the error and shows the safe answer with
  `guardrail_failed=true` in the trace.
- **Rationale**: Deterministic, cheap, and testable. It enforces Constitution I in code, not only in the prompt.
- **Alternatives**: A string (LLM-judged) guardrail, which costs an extra model call and is not deterministic. It is
  listed as a next experiment in the evaluation report.

## R7. Prompt-injection handling (Clarification 2)

- **Decision**: Each retrieved chunk is screened with a small set of regular expressions ("ignore (all|previous)
  instructions", "you are now", "system prompt", "disregard ...", "reveal ... key"). Matching chunks are kept,
  wrapped in `<<<EVIDENCE id=...>>> ... <<<END EVIDENCE>>>` like every chunk, and recorded as `injection_flags` in
  the retrieval span and the trace metadata. Both agents' prompts state that evidence is data and that instructions
  inside it must never be followed.
- **Rationale**: Matches the clarification. Regular expressions give a measurable detection rate on the evaluation's
  injection case, and they are honest about their limits (reported in the evaluation report).

## R8. Observability (Langfuse 5.2 + OpenInference)

- **Decision**:
  - `langfuse.Langfuse(public_key, secret_key, base_url)` is the client. `LANGFUSE_BASE_URL` is the variable named
    by the official integration page; `LANGFUSE_HOST` is also accepted by the SDK.
  - `CrewAIInstrumentor().instrument(skip_dep_check=True)` and `OpenAIInstrumentor().instrument()` are called once,
    before any crew is built. The OpenAI instrumentor is what records model name and token usage, because CrewAI
    calls OpenAI through the native SDK (R2).
  - One root observation per question: `start_as_current_observation(as_type="agent", name="rag-question",
    input={...})` inside `propagate_attributes(session_id=..., tags=["agentic-rag"], metadata={"request_id": ...},
    trace_name="rag-question")`.
  - Child observations:
    - `as_type="retriever"`: chunk ids, sources, scores, injection flags; snippets only, not full chunk text
    - `as_type="guardrail"`: pass or fail, and the reason
    - `as_type="span"`: ingestion
  - Errors are recorded with `level="ERROR"` and a `status_message`.
  - The trace id comes from `get_current_trace_id()`, and the link from `get_trace_url()`.
  - `flush()` runs after each question, because Streamlit reruns are short-lived.
- **Verified**:
  - Langfuse 5.x exports only spans that pass `is_default_export_span`. The `openinference` scope is on its
    allowlist, so CrewAI and OpenAI spans are exported.
  - `update_current_trace` no longer exists in 5.x. Trace-level attributes go through `propagate_attributes`, and
    trace input/output through the root observation.
- **Disabled mode**: When `LANGFUSE_ENABLED=false` or a key is missing, `NoopTracer` provides the same interface and
  does nothing (FR-018).
- **Privacy**:
  - The `session_id` (one per browser session) and the `request_id` (one per question) are random UUID4 values.
  - The OpenAI key is never put in metadata.
  - `CREWAI_DISABLE_TELEMETRY=true` and `CREWAI_TRACING_ENABLED=false` turn off CrewAI's own anonymous telemetry and
    its cloud tracing prompt. `OTEL_SDK_DISABLED` is **not** used, because it would also turn off the Langfuse
    exporter.

## R9. UI

- **Decision**: Streamlit, a single page with `st.tabs(["Knowledge", "Ask"])`.
  - The Knowledge tab uses `st.file_uploader(accept_multiple_files=True)`, a per-file status table, a knowledge base
    summary, and a clear button.
  - The Ask tab uses `st.status` while the crew runs, a status badge, the answer, and citation expanders.
- **Rationale**: Same UI stack as Homework 2, so the patterns are reused.

## R10. Evaluation

- **Decision**:
  - `evaluation/questions.jsonl` holds `id`, `question`, and `category`.
  - `evaluation/expected_answers.jsonl` holds `id`, `expected_status`, `expected_answer_points`, and
    `expected_sources`.
  - `scripts/run_evaluation.py` runs every question through the real crew and computes the metrics it can measure
    deterministically:
    - retrieval: Hit Rate@k, Recall@k, Precision@k, MRR, nDCG@k
    - citations: citation correctness and completeness
    - answers: status accuracy (abstention quality), answer-point coverage by keyword match
    - runs: latency, token usage, injection flag rate
  - Results, with trace ids, go to `evaluation/results/latest.json`.
  - `scripts/export_langfuse_traces.py` reads the real traces back through the Langfuse API.
- **Rationale**: The `rag-evaluation-report` skill must only report measured values (Constitution VII). These files
  are its sources.
- **Not measured automatically**: faithfulness and answer relevance (need an LLM judge or a human), and human
  preference. The report labels them unavailable and explains how to collect them, using the LLM-as-a-judge
  calibration from Homework 3, slide 11.
