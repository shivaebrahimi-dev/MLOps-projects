# Langfuse trace evidence

Real traces sent to Langfuse Cloud (US region, `https://us.cloud.langfuse.com`) on 2026-11-22. The trace ids below
were returned by the application, and `scripts/export_langfuse_traces.py` read them back through the Langfuse API.
The summaries are in [langfuse_traces_summary.json](langfuse_traces_summary.json).

## Live test (one question)

| Field | Value |
|---|---|
| Question | Which MinIO bucket stores the Fashion-MNIST model files? |
| Trace id | `e58cfe310c3707c958c6b30a8d97c816` |
| Observations | 12: root `rag-question` (agent) → CrewAI crew (chain) → Knowledge Retriever and Answer/Verifier agents → 2 OpenAI chat generations, `search_knowledge_base` tool, `retrieve` retriever, embedding, `citation-guardrail` |
| Model / tokens / cost | `gpt-4o-mini-2024-07-18` / 2,025 tokens / $0.000342 (as reported by Langfuse) |

## Evaluation run `eval-20261122-183719-9fbcef`

All 8 traces share the Langfuse session id `eval-20261122-183719-9fbcef`, so in Langfuse they can be opened together
under **Sessions**.

| Question | Trace id |
|---|---|
| q1 | `1b6589d23d28626f028f741bbbf20607` |
| q2 | `5d42962a4fe3cc0e16499d4ad8ac6c5f` |
| q3 | `76e370e29124fcd6f8ca63a134192378` |
| q4 | `c6d6133c7d06a27d806bd5ab841dca76` |
| q5 | `4280c83c57729f62c6b3d101dc77ee36` |
| q6 | `d07e5659ca3dd782150e334342a4c1c9` |
| q7 | `880f654b7c8af6784e1a95ab0025f82d` |
| q8 | `238f3611e542a8741dacb0b01fd54601` |

The trace ids are only identifiers: opening them requires access to the author's Langfuse project. The screenshot
[langfuse-trace-screenshot.png](langfuse-trace-screenshot.png) shows one trace as it appears in the Langfuse UI.
