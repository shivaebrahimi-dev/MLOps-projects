# UI Contract (Streamlit, `app/ui.py`)

The page header shows the app name, the model name, and a tracing badge: "Langfuse: on (base URL)" or
"Langfuse: off". It never shows keys.

## Tab 1: Knowledge

| Element | Behaviour | Requirement |
|---|---|---|
| File uploader (multiple; `.txt`, `.md`, `.pdf`) | The server-side check rejects other types too, not only the picker filter | FR-001, FR-002 |
| "Index files" button | Indexes the selected files with a spinner | FR-004 |
| Results table | One row per file: name, status (Indexed / Rejected), chunks, message | FR-004 |
| Summary message | Success: "Indexed N file(s), M chunk(s)". Error: "K file(s) rejected" | FR-004 |
| Knowledge base panel | Lists indexed sources with chunk counts and indexing time | FR-006 |
| "Clear knowledge base" button | Asks for confirmation, then deletes every chunk | FR-006 |

## Tab 2: Ask

| Element | Behaviour | Requirement |
|---|---|---|
| Empty-knowledge notice | When the knowledge base is empty, shows "Upload files in the Knowledge tab first" and disables the input | US2 scenario 4 |
| Question box and "Ask" button | Empty questions cannot be submitted; questions over 1000 characters are cut, with a notice | Edge cases |
| Processing state | A status box, "Agents are working…", while the crew runs | FR-008 |
| Status badge | Answered / Partial / Not found / Error | FR-012 |
| Answer | Markdown, with inline `[source#n]` citations | FR-010 |
| Missing evidence | Bullet list, shown for Partial answers | Clarification 1 |
| Sources | One expander per citation: file, chunk number, page, score, passage | FR-013 |
| Run details | Expander: latency, tokens, model, guardrail attempts, injection flags, trace id and link | FR-019 |
