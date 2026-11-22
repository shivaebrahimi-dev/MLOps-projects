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

## Evaluation run `eval-20261122-191648-447fd1` (current defaults: gpt-4o-mini, TOP_K=6)

All 8 traces share the Langfuse session id `eval-20261122-191648-447fd1`, so in Langfuse they can be opened together under
**Sessions**. Earlier runs (the TOP_K=4 baseline and three experiments) are also in Langfuse. Their exports are in
[experiments/](experiments/), and their session ids are the run ids in `evaluation/results/comparison.json`.

| Question | Trace id |
|---|---|
| q1 | `d8f1cf20fae14c4f1a94f9920fb4188f` |
| q2 | `6f4c46097c2dcaf2546d92bea84861f1` |
| q3 | `860397bfbcbabf25f5cae8ee1648901b` |
| q4 | `5771c741d2f51d48ba6caeb0f9aa5195` |
| q5 | `79737bf3cd75e1fb6ec6fb2996755e7c` |
| q6 | `b9fecd384f7083b16499e4cd6b103990` |
| q7 | `fbb39dd48be5073fd153a9dc822e21b9` |
| q8 | `356237b0128d105f4e66648a57d4430c` |

The trace ids are only identifiers: opening them requires access to the author's Langfuse project. The screenshot
[langfuse-trace-screenshot.png](langfuse-trace-screenshot.png) shows one trace as it appears in the Langfuse UI.

Note on cost: OpenAI caches repeated prompt prefixes automatically, and cached input tokens are cheaper. Two runs
with the same token count can therefore report different costs (for example $0.0025 vs $0.0032 for the two
TOP_K=6 runs, where the first had 10,368 cached input tokens).
