# Data Model: Agentic RAG Knowledge Assistant

Phase 1 of `/speckit-plan`. These are Python models in `app/models.py`: pydantic models where CrewAI or JSON
serialisation needs them, dataclasses otherwise. Entity names match [spec.md](spec.md) Key Entities.

## Document → stored as chunk metadata (no separate record)

Uploaded files are processed in memory and never written to disk (FR-007, Clarification 3). A document exists only
through its chunks. `KnowledgeBase.list_sources()` rebuilds the document list from chunk metadata.

| Field | Type | Rule |
|---|---|---|
| `source` | str | Original file name; the unique key. Re-uploading the same name replaces the old chunks (FR-005) |
| `file_type` | `txt`, `md` or `pdf` | From the extension, lower-cased (FR-001) |
| `sha256` | str | Hash of the uploaded bytes |
| `size_bytes` | int | Must be > 0 and ≤ `MAX_UPLOAD_MB` × 1 MiB (FR-002) |
| `indexed_at` | ISO-8601 str (UTC) | Set at ingestion |
| `chunk_count` | int | ≥ 1 |

## Chunk

| Field | Type | Rule |
|---|---|---|
| `chunk_id` | str | `{source}#{chunk_index}`, unique in the collection; also the Chroma id |
| `source` | str | As above |
| `chunk_index` | int | 0-based, in document order |
| `page` | int or None | 1-based page for PDFs; None otherwise. Chunks never cross pages |
| `text` | str | Not empty after stripping; length ≤ `CHUNK_SIZE` plus a little slack |
| `embedding` | list[float] | From the configured `Embedder`; stored only in Chroma |
| document fields | | `file_type`, `sha256`, `size_bytes`, `indexed_at` copied to every chunk's metadata |

## RetrievedChunk (evidence)

| Field | Type | Rule |
|---|---|---|
| `chunk` | Chunk | |
| `score` | float | `1 - cosine_distance`; higher is more relevant |
| `rank` | int | 1-based |
| `is_evidence` | bool | `score >= MIN_RELEVANCE` |
| `injection_flag` | bool | Set by `guardrails.detect_injection(text)` (FR-014) |

## IngestResult (one per uploaded file)

| Field | Type | Rule |
|---|---|---|
| `filename` | str | |
| `status` | `indexed` or `rejected` | |
| `chunks` | int | 0 when rejected |
| `message` | str | Success summary or the specific rejection reason (FR-002, FR-004) |

## AnswerDraft (output of the Answer/Verifier task, `output_pydantic`)

| Field | Type | Rule |
|---|---|---|
| `status` | `answered`, `partial` or `not_found` | Clarification 1 / FR-012 |
| `answer` | str | Inline citations `[source#n]` for every claim; equals `SAFE_ANSWER` when `not_found` |
| `citations` | list[str] | Chunk ids. Must be a subset of the retrieved ids. Not empty unless `not_found`, where it must be empty |
| `missing_evidence` | list[str] | Required (not empty) when `partial`; optional otherwise |

`SAFE_ANSWER = "I could not find this in the uploaded knowledge."` The formal shape is in
[contracts/answer.schema.json](contracts/answer.schema.json).

## AnswerResult (what the UI and the evaluation script receive)

| Field | Type | Notes |
|---|---|---|
| `question` | str | Trimmed; at most `MAX_QUESTION_CHARS` (1000) |
| `status` | `answered`, `partial`, `not_found` or `error` | `error` only for failed runs (FR-016) |
| `answer` | str | |
| `citations` | list[RetrievedChunk] | Resolved from the draft's ids |
| `retrieved` | list[RetrievedChunk] | Everything the tool returned (for evaluation) |
| `missing_evidence` | list[str] | |
| `injection_flags` | list[str] | Chunk ids flagged |
| `guardrail` | passed, attempts, reasons, fallback | |
| `latency_s` | float | End to end |
| `token_usage` | prompt, completion, total, requests (or None) | From `CrewOutput.token_usage` |
| `model` | str | For example `gpt-4o-mini` |
| `trace_id`, `trace_url` | str or None | None when tracing is off |
| `request_id` | str | UUID4 |

## State transitions

Upload:

1. received → **validate**. On failure → `rejected` (with reason).
2. parsed → chunked → embedded → upserted → `indexed`.

Answer:

1. question → retrieving → drafting → guardrail check.
2. Guardrail passes → `answered`, `partial`, or `not_found`.
3. Guardrail fails → feedback to the agent and a retry (at most 2 retries).
4. Retries used up → `not_found` with `guardrail.fallback = true`.
5. Any exception → `error`, and the trace's root observation gets level ERROR.

## Evaluation item and result

See [contracts/evaluation-dataset.md](contracts/evaluation-dataset.md).
