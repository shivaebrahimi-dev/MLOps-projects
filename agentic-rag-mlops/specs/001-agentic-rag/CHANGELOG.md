# Specification changelog: 001-agentic-rag

This file is the evidence of specification iteration required by the assignment (deliverable 2). Each entry links a
spec change to its commit, so the history can be checked with `git log -- specs/`.

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
