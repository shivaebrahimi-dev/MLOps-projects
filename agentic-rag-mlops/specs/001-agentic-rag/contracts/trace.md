# Langfuse Trace Contract

One trace per question (FR-015). The names of our own observations are fixed. The OpenInference names come from the
instrumentors and may differ slightly between versions.

Expected tree:

- `rag-question` (as_type `agent`). Input: question and request_id. Output: status, answer, citations.
  - CrewAI crew kickoff (OpenInference, CHAIN)
    - Retriever agent and task (OpenInference, AGENT)
      - OpenAI chat completion (OpenInference, LLM): model name and token usage
      - `search_knowledge_base` tool (OpenInference, TOOL)
        - `retrieve` (as_type `retriever`). Output: chunk_id, source, page, score, rank, snippet for each chunk.
          Metadata: top_k, min_relevance, evidence_count, injection_flags.
    - Answer/Verifier agent and task (OpenInference, AGENT)
      - OpenAI chat completion (LLM)
      - `citation-guardrail` (as_type `guardrail`). Output: passed, reasons, attempt.

Trace attributes come from `propagate_attributes`:

- `session_id`: a UUID4 per browser session
- `trace_name`: `rag-question`
- `tags`: `agentic-rag`
- `metadata`: request_id, app_version, model, top_k

No `user_id` is set, because there is one anonymous local user.

On failure, the root observation gets `level="ERROR"` and a `status_message` with the exception type and a message
that contains no secrets.

Never in a trace: API keys, `.env` content, user names or emails, whole uploaded files.
