# Evaluation Dataset Contract

Both files are JSON Lines (one JSON object per line) and are joined on `id`.

## `evaluation/questions.jsonl`

```json
{"id": "q1", "question": "...", "category": "answerable"}
```

| Field | Type | Values |
|---|---|---|
| `id` | str | `q1`, `q2`, ... |
| `question` | str | |
| `category` | str | `answerable`, `unanswerable`, `multi_chunk`, `partial`, `injection` |

FR-021 minimum: at least 3 `answerable`, 1 `unanswerable`, and 1 `multi_chunk`.

## `evaluation/expected_answers.jsonl` (the rubric)

```json
{"id": "q1", "expected_status": "answered", "expected_sources": ["fashion-mnist-pipeline.md"],
 "expected_answer_points": [{"point": "Uses a LeNet-5 CNN", "keywords": ["lenet"]}], "notes": "..."}
```

| Field | Type | Meaning |
|---|---|---|
| `expected_status` | `answered`, `partial` or `not_found` | Used for abstention quality |
| `expected_sources` | list[str] | Relevant documents: the ground truth for the retrieval metrics |
| `expected_answer_points` | list of {point, keywords} | Rubric. A point counts as covered when any of its keywords appears in the answer (case-insensitive). This is a keyword proxy, not a judge of meaning |
| `notes` | str | Why the item exists |

## `evaluation/results/latest.json` (written by `scripts/run_evaluation.py`)

Top-level fields:

- `run_id`, `started_at`, `finished_at`
- `config`: model, embedding model, top_k, min_relevance, chunk_size, chunk_overlap
- `tracing_enabled`
- `ingestion`: files, chunks, rejected, latency_s
- `items`: one entry per question (below)
- `summary`: means of each metric over the items it applies to, plus counts

Each item has:

- `id`, `category`, `question`, `status`, `expected_status`
- `retrieved`: chunk_id, source, score, rank for each retrieved chunk
- `citations`, `answer`, `points_covered`, `points_missed`
- `metrics`: hit@k, recall@k, precision@k, mrr, ndcg@k, citation_correctness, citation_completeness,
  status_correct, answer_point_coverage
- `latency_s`, `token_usage`, `guardrail`, `injection_flags`, `trace_id`, `error`

## Metric definitions

Relevance is judged at **document** level: a retrieved chunk is relevant if its `source` is in `expected_sources`.

- **Hit@k** = 1 if any of the top-k chunks is relevant, else 0.
- **Recall@k** = relevant documents found in the top k / expected documents.
- **Precision@k** = relevant chunks in the top k / number of chunks returned.
- **MRR** = 1 / rank of the first relevant chunk (0 if none).
- **nDCG@k** uses binary relevance per chunk.
- **Citation correctness** = cited ids that were retrieved **and** come from an expected source / cited ids.
- **Citation completeness** = expected sources that were cited / expected sources.
- **Status correct** = `status == expected_status`.

Retrieval and citation metrics apply only to items with a non-empty `expected_sources`.
