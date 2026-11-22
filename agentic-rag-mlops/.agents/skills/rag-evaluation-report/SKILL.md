---
name: rag-evaluation-report
description: Write an honest evaluation report for this project's agentic RAG assistant from its specs, test results, evaluation dataset and results, configuration, and exported Langfuse trace summaries. Use only when the user explicitly asks for the RAG evaluation report.
user-invocable: true
disable-model-invocation: true
argument-hint: "[optional output path, default reports/rag-evaluation-report.md]"
allowed-tools: Read, Glob, Grep, Write
---

# RAG evaluation report

This skill writes `reports/rag-evaluation-report.md`, an evaluation report for the Agentic RAG Knowledge
Assistant in this folder. If the user gives a different output path as the argument, use that path instead.

## When this skill runs

- **Only when the user explicitly asks**, for example with `/rag-evaluation-report` or "write the RAG evaluation
  report". The frontmatter sets `disable-model-invocation: true`, so the agent never starts it on its own.
- It does **not** run during coding, uploads, or question answering. The application never calls it.
- It is meant to run **after** the application is implemented and tested. Normally the user has already run:

  ```bash
  uv run pytest -q -rs > reports/test-results.txt
  uv run python scripts/run_evaluation.py
  uv run python scripts/export_langfuse_traces.py
  ```

  If any of these outputs is missing, still write the report. Mark the affected metrics **Unavailable** and give
  the command that produces them. Do not run the commands yourself: this skill only reads files and writes the
  report.

## Ground rules (non-negotiable)

1. **Never invent a value.** Every number in the report must come from one of the input files, or be computed from
   them with the formula written next to it. This applies above all to Langfuse data: latency, tokens, cost, and
   trace counts come **only** from `reports/langfuse_traces_summary.json` or `evaluation/results/latest.json`. If a
   value is missing, write `Unavailable` and explain how to collect it.
2. **Label every metric** with exactly one status:
   - `Measured`: copied from a results file (name the file and the JSON key).
   - `Derived`: computed by this skill from measured values (show the formula and inputs).
   - `Proxy`: a measured stand-in for a metric that needs a judge or a human (say what it stands in for and why it
     is weaker).
   - `Unavailable`: not collected yet (say how to collect it).
   - `Not applicable`: does not apply to this system (say why).
3. **Copy numbers exactly** as they appear in the files, rounded to at most 3 decimals. Always give n (how many
   items a mean covers). With n < 10, say that the sample is too small for firm conclusions.
4. **Privacy:** never open `.env` or any file that may hold secrets. Read configuration only from `.env.example`,
   `app/config.py`, and the `config` block of the results file. Do not copy full chunk texts, API keys, or personal
   data into the report. Short quotes from answers are fine.
5. **Stale inputs:** if `langfuse_traces_summary.json` has an `evaluation_run_id` that differs from the `run_id` in
   `evaluation/results/latest.json`, or the test results are older than the code changes you can see, say so in
   the report and do not mix numbers from different runs in one table without labelling them.

## Inputs

Read these files, relative to the project root. Record in the report which ones you found.

| Input | Path | Used for |
|---|---|---|
| Specification | `specs/001-agentic-rag/spec.md`, `specs/constitution.md` | Success criteria SC-001 to SC-009, requirements |
| Plan and research | `specs/001-agentic-rag/plan.md`, `research.md` | System description, known design limits (R4, R6, R7) |
| Spec changes | `specs/001-agentic-rag/CHANGELOG.md` | Specification iterations |
| Application configuration | `.env.example`, `app/config.py` | Defaults: model, embedding model, top_k, MIN_RELEVANCE, chunking |
| Test results | `reports/test-results.txt` | Passed, failed, and skipped counts; what the tests cover |
| Evaluation dataset | `evaluation/questions.jsonl`, `evaluation/expected_answers.jsonl` | Questions, categories, rubric |
| Evaluation results | `evaluation/results/latest.json` | All measured quality, latency, and token values |
| Langfuse trace summaries | `reports/langfuse_traces_summary.json` | Trace completeness, step latency, tokens, cost, errors |
| Experiments (optional) | `evaluation/results/comparison.json`, `evaluation/results/experiments/*.json`, `reports/experiments/langfuse_*.json` | Baseline vs other models or settings; each row's values are copied from its own run files |
| Trace evidence | `reports/` (a screenshot or a trace id file, if present) | Proof that a real trace exists |
| Metric definitions | `specs/001-agentic-rag/contracts/evaluation-dataset.md` | Exact formulas |

## Procedure

1. **Inventory.** List every input with found or missing, and its run id or timestamp: `run_id`, `finished_at`,
   `fetched_at`, the date line in the test results.
2. **Load and cross-check.**
   - The dataset has the same number of items as `latest.json → items`.
   - `latest.json → config` matches the defaults in `app/config.py`. Report any difference: it means the run used
     non-default settings.
   - The export's `evaluation_run_id` matches `latest.json → run_id`.
   - Count passed, failed, and skipped tests in `reports/test-results.txt`.
3. **Fill the metric catalogue** below, one row per metric, using the sources listed for each.
4. **Check the success criteria** SC-001 to SC-009 from the spec: met, not met, or not verifiable from the files,
   each with its evidence.
5. **Secret scan.** Search `evaluation/results/`, `reports/`, and the report draft for `sk-`, `pk-lf-`, `sk-lf-`,
   and `Bearer `. Report the outcome as a measured safety check. A hit is a finding: name the file, not the secret.
6. **Write the report** with the template below.
7. **Self-check before saving.** Every number in the report appears in an input file, or is marked Derived with
   its formula. No row says Measured without naming a source. Fix anything that fails.

## Metric catalogue

The basis is the author's Homework 3 presentation on evaluation metrics:

- **Slide 9, "What to measure in an LLM or RAG system"**: faithfulness, answer relevance, retrieval quality,
  hallucination rate, safety, cost and latency.
- **Slide 10, "Automated, human, or LLM-as-a-Judge?"**: automated checks for CI, a judge for scale, humans for gold
  labels.
- **Slide 11, "Trust an LLM judge only after checking it"**: calibrate a judge against human labels with raw
  agreement and Cohen's κ, and watch for leniency, position, verbosity, and self-preference bias.
- **Slide 12** (tail latency, SLOs) and **slide 14** (monitoring) for the operations rows.

Abbreviations in the "Where to find it" column: `R` = `evaluation/results/latest.json`, `S` = its `summary`,
`L` = `reports/langfuse_traces_summary.json`.

### Retrieval

| Metric | Where to find it | Usual status |
|---|---|---|
| Hit Rate@k | `S.metrics["hit@k"]` | Measured |
| Recall@k (document level) | `S.metrics["recall@k"]` | Measured |
| Precision@k | `S.metrics["precision@k"]` | Measured |
| MRR | `S.metrics.mrr` | Measured |
| nDCG@k (binary) | `S.metrics["ndcg@k"]` | Measured |
| Context relevance | Not computed. Proxy: `R.score_distribution` (relevant vs irrelevant chunk scores) | Proxy. Full metric needs an LLM judge or a human rating each retrieved chunk |
| Chunk coverage | Derived: distinct retrieved chunk ids over all items ÷ `R.ingestion.chunks` | Derived |
| Threshold check | `R.score_distribution`: is `MIN_RELEVANCE` between the relevant and irrelevant medians? Is it above `top_score_on_unanswerable`? | Derived |

### Generation and RAG quality

| Metric | Where to find it | Usual status |
|---|---|---|
| Faithfulness / groundedness | Not computed | Unavailable. Collect with an LLM judge that scores each claim against the cited chunks (for example RAGAS faithfulness), calibrated against ~30 human labels with Cohen's κ (slide 11) |
| Answer relevance | Not computed | Unavailable. Same approach: judge or human rating, 1 to 5 |
| Context utilisation | Derived: mean over answered items of (cited chunks ÷ evidence chunks retrieved) | Derived |
| Citation correctness | `S.metrics.citation_correctness` | Measured |
| Citation completeness | `S.metrics.citation_completeness` | Measured |
| Unsupported-claim rate | Proxy: guardrail rejections (`S.guardrail`, `items[].guardrail.attempts > 1`) count drafts that cited chunks they were not given, or nothing at all | Proxy. A true rate needs claim-level judging |

### Task outcome

| Metric | Where to find it | Usual status |
|---|---|---|
| Correctness against expected answer points | `S.metrics.answer_point_coverage`, plus per item `points_covered` and `points_missed` | Proxy: keyword matching, not meaning |
| Abstention quality | `S.abstention`, `S.status_accuracy`, per-item `status` vs `expected_status` | Measured |
| Partial-answer handling | Items with `category = partial`: status, `missing_evidence` | Measured |
| Human preference or rubric score | Not collected | Unavailable. Have one or two people grade each answer with the rubric in `expected_answers.jsonl`, and report their agreement |

### Agent workflow

| Metric | Where to find it | Usual status |
|---|---|---|
| Task success rate | Derived: (items − `S.errors`) ÷ items | Derived |
| Tool / retrieval failure rate | `S.errors`; items with an empty `retrieved` list | Measured |
| Number of retries | `S.guardrail.mean_attempts`, `S.guardrail.fallbacks`, `first_try_pass_rate` | Measured |
| Trace completeness | `L.aggregate`: `with_retriever_observation`, `with_guardrail_observation`, `with_token_usage`, `with_model_name` against `traces`; `L.missing` | Measured, if L exists |
| Step latency | `L.traces[].step_latency_s_by_type` (GENERATION, TOOL, RETRIEVER, ...) | Measured, if L exists |
| End-to-end latency | `S.latency_s` (median, p95, max) and `L.aggregate.latency_s` | Measured. Compare with SC-007 (median < 30 s) |
| Token usage and cost | `S.tokens` (CrewAI counts); `L.aggregate.total_tokens`, `L.aggregate.total_cost_usd` (Langfuse) | Measured. Cost is Unavailable if Langfuse returned no cost |

### Operations

| Metric | Where to find it | Usual status |
|---|---|---|
| Upload success rate | `R.ingestion` (indexed ÷ files) for valid files; the rejection tests in `reports/test-results.txt` for invalid files | Measured |
| Indexing latency | `R.ingestion.latency_s` (all sample files, including embedding calls) | Measured |
| Query throughput | Derived: items ÷ (`finished_at` − `started_at`), single user, sequential | Derived. Not a load test |
| Error rate | `S.errors` ÷ `S.items`; `L.aggregate.with_error_level` | Measured |
| Resource usage (CPU, memory, disk) | Not collected | Unavailable. Collect with `psutil` sampling during the run, or OS monitoring |

### Safety and privacy

| Metric | Where to find it | Usual status |
|---|---|---|
| Prompt-injection detection rate | `S.injection` (flagged ÷ injection items, false flags); the injection tests in the test results | Measured (tiny n) |
| Injection resistance | Item with `category = injection`: did the answer follow the planted instruction? (for example, does it say the course is cancelled?) | Measured, by reading the answer |
| Data leakage | Do answers quote content that is not in their citations? Proxy: citation correctness | Proxy |
| Secret exposure | Procedure step 5 scan, plus `test_traces_contain_no_secrets` and the redaction test | Measured |
| PII handling | Design evidence: random session and request ids, no user id, uploads not stored (spec FR-007, FR-017; research R8) | Not measured. Describe the design, and recommend a PII scan of traces |

## Report template

Use these headings in this order. Keep tables compact. Write for a course evaluator who knows ML but has not seen
the code.

```markdown
# RAG Evaluation Report: Agentic RAG Knowledge Assistant

Generated by the `rag-evaluation-report` skill on <date>, at the user's explicit request.
Evaluation run: `<run_id>` · Model: `<model>` · Embeddings: `<embedding_model>` · Traces: <found>/<requested>

## 1. Summary
3 to 6 bullets with the most important measured findings, the biggest gaps, and whether the success criteria are met.

## 2. System under evaluation
Two-agent sequential CrewAI crew, retrieval settings, guardrail, tracing. Taken from plan.md and the run config.

## 3. Evidence used
Table: input, path, found or missing, run id or timestamp.

## 4. Automated tests
Passed, failed, and skipped counts, and what the suite covers (FR-020). Quote the summary line from the file.

## 5. Metrics
One table per family (Retrieval, Generation and RAG quality, Task outcome, Agent workflow, Operations, Safety and
privacy). Columns: Metric | Value | n | Status | Source | Notes, or how to collect.

## 6. Results per question
Table: id | category | expected → actual status | hit@k | citations | points covered | latency | trace id.

## 6b. Experiments (only if comparison.json exists)
Table copied from comparison.json: configuration, status accuracy, citation correctness and completeness,
precision@k, latency, tokens, Langfuse cost. Say what changed between runs (model or TOP_K), that each
configuration ran once on a small dataset, and which differences are smaller than one question (1/n).

## 7. Langfuse traces
What the export shows: observation types per trace, completeness, step latency, tokens, cost, errors, missing
traces. If the export is missing: "Unavailable: run scripts/export_langfuse_traces.py".

## 8. Success criteria
SC-001 to SC-009: met, not met, or not verifiable, with evidence.

## 9. Limitations
At least: small n; synthetic, self-written knowledge; keyword proxy for correctness; document-level relevance
labels; no judge or human ratings; single run (LLM non-determinism); regex injection screen.

## 10. Data-quality concerns
For example: the questions were written by the same person who wrote the documents (optimistic bias); the glossary
overlaps with other documents (a distractor); the injection test document is synthetic; few chunks per document.

## 11. Privacy and security
What is protected and how (secrets, PII, uploads, telemetry), plus the result of the secret scan.

## 12. Recommended next experiments
Concrete and prioritised, for example: an LLM judge for faithfulness and relevance, calibrated with κ against human
labels (HW3 slide 11); sweep top_k and MIN_RELEVANCE using the score distribution; a larger dataset with hard
negatives; repeated runs to measure variance; resource monitoring; a regression gate in CI on hit@k and status
accuracy.

## Appendix: metric definitions
Short formulas, and the mapping to HW3 slides 9 to 12 and 14.
```

After saving, tell the user where the report is, how many metrics were Measured, Derived, Proxy, and Unavailable,
and which missing inputs (if any) would fill the gaps.
