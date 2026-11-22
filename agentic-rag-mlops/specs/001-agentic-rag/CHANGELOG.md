# Specification changelog: 001-agentic-rag

This file is the evidence of specification iteration required by the assignment (deliverable 2). Each entry links a
spec change to its commit, so the history can be checked with `git log -- specs/`.

## Iteration 3 — 2026-11-22 — `/speckit-analyze` remediation

A cross-artifact analysis of the constitution, spec, plan, and tasks found no critical issues and no constitution
violations. Coverage: 24/24 functional requirements have tasks. It found six small gaps, and the user approved fixing
all of them.

| ID | Severity | Finding | Change |
|---|---|---|---|
| A1 | Medium | The edge case "same file name twice in one upload" had no task | Added to T013 and T014 |
| A2 | Medium | Long questions were truncated only in the UI, but the evaluation script bypasses the UI | T020 truncates in `answer_question()` |
| A3 | Medium | research.md said the threshold is set from evaluation, but no task did it | T037 records the score distributions |
| A4 | Medium | Nothing checked that token usage and model name reach the traces | T038 reports whether they arrived |
| A5 | Low | SC-007 "typical question" was vague | SC-007 now uses the median latency over the evaluation run |
| A6 | Low | The edge case "vector store deleted while running" had no task | Added to T009 |

## Iteration 2 — 2026-11-22 — `/speckit-clarify`

An ambiguity scan of spec v1 found three decisions that change behaviour and tests. The user answered them, and the
spec was updated (see its `## Clarifications` section).

| # | Question | Decision | Spec changes |
|---|---|---|---|
| 1 | What should happen when the documents answer only part of a question? | Answer the supported part with citations, mark it "Partial", and list the missing evidence. Refuse only when nothing relevant is retrieved. | FR-012 now defines three answer statuses (Answered, Partial, Not found); new US3 scenario 4; Answer entity gains `status`. |
| 2 | What should happen when a retrieved document contains instructions aimed at the assistant? | Keep it as quoted evidence, never follow it, flag the run in the trace. | FR-014 requires delimited quoted evidence and an injection flag; US3 scenario 3 reworded; Answer entity gains `injection flag`. |
| 3 | Should original uploaded files be kept? | No. Keep only chunks, metadata, and a content hash. | FR-007 forbids saving uploads; Document entity notes the file is not kept. |

Checklist: all items still pass.

## Iteration 1 — 2026-11-22 — `/speckit-specify`

First version: 5 user stories (upload and index, cited answers, safe abstention, tracing, evaluation report), 24
functional requirements, 9 success criteria, edge cases and assumptions. Constitution v1.0.0 ratified the same day.
