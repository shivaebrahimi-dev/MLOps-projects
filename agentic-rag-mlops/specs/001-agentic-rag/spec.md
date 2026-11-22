# Feature Specification: Agentic RAG Knowledge Assistant

**Feature Branch**: `add-agentic-rag-assistant`

**Created**: 2026-11-22

**Status**: Clarified (iteration 2)

**Input**: User description: "Build and test a small, production-minded RAG application. A user uploads knowledge
files, the application indexes them, and the user asks questions answered only from those files, with citations.
Answering is done by an agentic workflow with a retriever agent and an answer/verifier agent, guarded against
hallucination. Every question is traced end to end in an observability platform. The project ships automated tests,
an evaluation dataset, and a user-invoked skill that writes an honest evaluation report." (Full text:
[homework04-assignment.md](../../homework04-assignment.md).)

## Clarifications

### Session 2026-11-22

- Q: When the documents answer only part of a question, what should the assistant do? → A: Answer the supported
  part with citations and add a "Missing evidence" note for the rest; refuse outright only when nothing relevant is
  retrieved.
- Q: If a retrieved document contains text that looks like instructions to the assistant, what should happen? → A:
  Keep it as evidence but present it to the answer agent as quoted data, never follow it, and flag the run in its
  trace.
- Q: Should the app keep a copy of the original uploaded files, or only the extracted chunks? → A: Only chunks,
  their metadata, and a content hash; uploaded files are processed in memory and not saved.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Upload and index knowledge files (Priority: P1)

A user opens the Knowledge tab, selects one or more files, and indexes them. For each file the user sees its name,
its processing status, and a clear success or error message. Valid files become searchable; invalid files are
rejected with a reason and nothing is stored for them.

**Why this priority**: Nothing can be answered until knowledge is indexed. This is the foundation of the product.

**Independent Test**: Upload the sample files and one invalid file. The valid files show "Indexed" with a chunk count,
the invalid file shows an error, and the knowledge base summary lists only the valid files.

**Acceptance Scenarios**:

1. **Given** an empty knowledge base, **When** the user uploads a `.txt`, a `.md` and a `.pdf` file with text,
   **Then** each file shows "Indexed" with its number of chunks, and a success message is shown.
2. **Given** any state, **When** the user uploads an empty file, a whitespace-only file, or a `.docx` file,
   **Then** that file shows "Rejected" with the reason ("empty file", "unsupported file type: .docx"), and the
   knowledge base is unchanged.
3. **Given** a file that is already indexed, **When** the user uploads the same file name again, **Then** its old
   chunks are replaced, not duplicated.
4. **Given** several files in one upload where one is invalid, **When** the user indexes them, **Then** the valid
   files are indexed and only the invalid one is rejected.

---

### User Story 2 - Ask a question and get a cited answer (Priority: P1)

A user opens the Ask tab, types a natural-language question, and submits it. While the agents work, the user sees a
processing indicator. The user then sees an answer written only from the indexed knowledge, with citations that name
the source file and chunk, and can expand each citation to read the supporting passage.

**Why this priority**: This is the core value of the product. Together with Story 1 it forms the MVP.

**Independent Test**: With the sample knowledge indexed, ask one of the answerable evaluation questions. The answer
contains the expected points and cites the expected source file.

**Acceptance Scenarios**:

1. **Given** indexed knowledge that contains the answer, **When** the user asks the question, **Then** a processing
   indicator is visible during the run and the answer is shown with at least one citation in the form
   `[file#chunk]`.
2. **Given** an answer with citations, **When** the user expands the sources, **Then** each citation shows the file
   name, chunk number, relevance score, and the passage text.
3. **Given** a question whose answer spans two documents, **When** the user asks it, **Then** the answer combines
   both and cites both documents.
4. **Given** an empty knowledge base, **When** the user opens the Ask tab, **Then** the user is told to upload files
   first and cannot submit a question.

---

### User Story 3 - Safe answer when the evidence is missing (Priority: P2)

When the indexed knowledge does not contain the answer, the user receives the safe response "I could not find this in
the uploaded knowledge", optionally with a note on what evidence is missing, instead of an invented answer.

**Why this priority**: Trustworthiness. A wrong confident answer is the most harmful failure of a RAG system.

**Independent Test**: Ask the unanswerable evaluation question. The safe response is shown with no citations.

**Acceptance Scenarios**:

1. **Given** indexed knowledge that does not cover the question, **When** the user asks it, **Then** the safe
   response is shown, marked "Not found", with no citations.
2. **Given** a draft answer that cites a chunk that was not retrieved, or states facts without any citation,
   **When** the hallucination guardrail checks it, **Then** the draft is rejected and either corrected or replaced
   by the safe response.
3. **Given** an indexed document that contains instructions aimed at the assistant (for example "ignore previous
   instructions"), **When** it is retrieved, **Then** the assistant treats it as quoted data, does not follow it,
   and the run is flagged in its trace.
4. **Given** indexed knowledge that covers only part of a question, **When** the user asks it, **Then** the answer
   gives the supported part with citations, is marked "Partial", and lists what evidence is missing.

---

### User Story 4 - Trace every question (Priority: P2)

A developer or evaluator opens the observability platform and finds one trace for each question. The trace shows the
question, a non-identifying session id, the agent and task steps, the retrieval step with the retrieved chunk
references, the model calls with model name and token usage, the guardrail result, latency, and status. Failed runs
are visible as errors. The Ask tab shows the trace id of each answer so the trace can be found.

**Why this priority**: Required for operating and evaluating the system, but the product works without it.

**Independent Test**: Ask one question with tracing configured, copy the trace id from the Ask tab, and open the
matching trace.

**Acceptance Scenarios**:

1. **Given** tracing is configured, **When** the user asks a question, **Then** exactly one trace appears, with
   nested steps for both agents, both tasks, retrieval, the model calls, and the guardrail.
2. **Given** tracing is disabled or its keys are missing, **When** the user asks a question, **Then** the answer
   still works and the UI states that tracing is off.
3. **Given** a run fails (for example, the model provider is unreachable), **When** the user asks a question,
   **Then** the user sees a clear error message and the trace is recorded with an error status.
4. **Given** any trace, **When** it is inspected, **Then** it contains no API keys and no personal data about the
   user.

---

### User Story 5 - Evaluate the system on request (Priority: P3)

After the application is implemented and tested, the user explicitly invokes the `rag-evaluation-report` skill. It
reads the specifications, the test results, the evaluation dataset and results, the configuration, and the available
trace summaries, and writes a Markdown report. The report lists the applicable metrics, gives only measured values,
labels missing values as unavailable with how to collect them, and covers limitations, data quality, privacy, and
next experiments.

**Why this priority**: Closes the MLOps loop, but depends on all other stories.

**Independent Test**: Run the evaluation, then invoke the skill. The report exists, and every number in it can be
traced to a results file.

**Acceptance Scenarios**:

1. **Given** evaluation results exist, **When** the user invokes the skill, **Then** a report is written to
   `reports/rag-evaluation-report.md`.
2. **Given** a metric with no data (for example, human preference), **When** the report is written, **Then** that
   metric is labelled "unavailable" with how to collect it.
3. **Given** normal use of the application (upload, ask) or of the coding agent, **When** no explicit request is
   made, **Then** the skill never runs.

### Edge Cases

- A PDF with no extractable text (a scanned image): rejected with "no extractable text".
- A file larger than the size limit: rejected with the limit in the message.
- A file name with spaces or non-ASCII characters: indexed, and shown correctly in citations.
- A very short question (for example, one word) or an empty question: an empty question is not submitted; a short
  one is answered or safely refused.
- A very long question: truncated to a maximum length, with a notice.
- The vector store folder is deleted while the app runs: the app reports an empty knowledge base, not a crash.
- The model returns output that is not in the expected structure: the guardrail treats it as a failure and the safe
  response is used.
- Two files with the same name in one upload: the later one replaces the earlier one, and the user is told.

## Requirements *(mandatory)*

### Functional Requirements

**Upload and indexing**

- **FR-001**: The system MUST accept `.txt`, `.md`, and `.pdf` files, several at a time.
- **FR-002**: The system MUST reject unsupported file types, empty or whitespace-only files, PDFs without
  extractable text, and files above the size limit, with a specific reason, and MUST NOT store anything for them.
- **FR-003**: The system MUST split each accepted document into overlapping chunks and store each chunk with its
  source file name, chunk number, a stable chunk id, and the page number for PDFs.
- **FR-004**: The system MUST show, per file, the name, status (Indexed or Rejected), number of chunks, and message.
- **FR-005**: Re-uploading a file with the same name MUST replace its previous chunks.
- **FR-006**: The system MUST show which files are currently in the knowledge base, and MUST let the user clear it.
- **FR-007**: The knowledge base MUST persist across application restarts. The system MUST store only chunks, their
  metadata, and a content hash; uploaded files MUST be processed in memory and MUST NOT be saved to disk.

**Question answering**

- **FR-008**: Users MUST be able to submit a natural-language question and see a processing indicator until the
  answer is ready.
- **FR-009**: Answers MUST be produced by a two-step agent workflow: a retriever step that collects the most relevant
  chunks with their source metadata, then an answer/verifier step that drafts the answer from those chunks only.
- **FR-010**: The answer step MUST be instructed to use only retrieved evidence, to separate evidence from
  uncertainty, to cite every claim as `[file#chunk]`, and to list missing evidence.
- **FR-011**: A hallucination guardrail MUST reject a draft that cites a chunk not in the retrieved set, or that is
  not an abstention but cites nothing. A rejected draft MUST be retried a limited number of times and then replaced by
  the safe response.
- **FR-012**: When no retrieved chunk reaches the relevance threshold, the system MUST return the safe response
  "I could not find this in the uploaded knowledge" without inventing content. When the evidence covers only part of
  the question, the system MUST answer the supported part with citations, mark the answer "Partial", and list the
  missing evidence. Every answer MUST have one status: Answered, Partial, or Not found.
- **FR-013**: The answer view MUST show the citations with file name, chunk number, relevance score, and passage.
- **FR-014**: Text inside documents MUST be treated as data. Retrieved chunks MUST be passed to the answer step as
  clearly delimited quoted evidence, with an instruction never to follow instructions found in them. Chunks that look
  like instructions to the assistant MUST be kept as evidence and flagged in the run's trace.

**Observability**

- **FR-015**: Each question MUST create one trace containing the question, a random session id and request id, nested
  agent, task, retrieval, model-call, and guardrail steps, latency, status, and the model name and token usage when
  available.
- **FR-016**: Failed runs MUST be recorded with an error status and message.
- **FR-017**: Traces, logs, and error messages MUST NOT contain secrets or personal data.
- **FR-018**: Tracing MUST be switchable off, and the application MUST work normally without tracing credentials.
- **FR-019**: The Ask tab MUST show the trace id of each answered question when tracing is on.

**Quality and evaluation**

- **FR-020**: Automated tests MUST cover file validation, chunking and metadata, retrieval of known evidence,
  cautious answers to unknown questions, citations on answerable questions, disabled or mocked tracing, and an
  end-to-end smoke test, all without network access.
- **FR-021**: The project MUST include an evaluation dataset of at least five questions: at least three answerable,
  one unanswerable, and one that needs more than one chunk or document, each with expected answer points and expected
  source documents.
- **FR-022**: The project MUST include a script that runs the evaluation dataset through the real workflow and saves
  per-question results, including retrieved sources, citations, abstention, latency, token usage, and trace ids.
- **FR-023**: The project MUST include the `rag-evaluation-report` skill, invoked only by explicit user request, that
  writes a Markdown report without inventing values (see User Story 5).
- **FR-024**: Secrets MUST be supplied through environment variables, with a `.env.example` listing variable names
  only.

### Key Entities

- **Document**: an uploaded file, known only through its chunks (the file itself is not kept). Attributes: file
  name, type, size, content hash, upload time, number of chunks.
- **Chunk**: a passage of a document. Attributes: chunk id (`file#n`), source file name, chunk number, page (PDF),
  text, embedding.
- **Retrieved evidence**: a chunk returned for a question, with its relevance score and rank.
- **Answer**: the response to a question. Attributes: text, status (Answered, Partial, Not found), citations (chunk
  ids), missing evidence notes, injection flag, guardrail result, trace id, latency.
- **Evaluation item**: a question with its category (answerable, unanswerable, multi-document), expected answer
  points, and expected source documents.
- **Evaluation result**: the outcome of running one evaluation item: retrieved sources, answer, citations,
  abstention, metrics, latency, tokens, trace id.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: An evaluator with the documented prerequisites can install, configure, start the UI, upload the sample
  files, and get a cited answer in under 15 minutes, following only the README.
- **SC-002**: 100% of invalid uploads in the test suite (empty, whitespace-only, unsupported type, no extractable
  text, too large) are rejected with a specific reason.
- **SC-003**: For every answerable evaluation question, at least one expected source document is among the retrieved
  chunks (Hit Rate@k = 100% on the dataset).
- **SC-004**: For every answerable evaluation question, the answer cites at least one chunk, and every citation
  refers to a retrieved chunk (citation correctness = 100%).
- **SC-005**: The unanswerable evaluation question receives the safe response in every run.
- **SC-006**: Every question asked with tracing on produces exactly one trace containing all required steps.
- **SC-007**: A typical question is answered in under 30 seconds.
- **SC-008**: The automated test suite passes with no network access and no API keys.
- **SC-009**: The evaluation report contains no metric value that cannot be traced to a results file.

## Assumptions

- Single local user. Authentication, multiple users, and access control are out of scope.
- The knowledge base is small (tens of documents, each up to 10 MB). Scaling and distributed storage are out of scope.
- English documents and questions.
- A hosted language model and embedding service are used; the user supplies their own API key.
- The observability platform is a hosted service; the user supplies their own project keys.
- Each question is answered independently. Conversation memory across questions is out of scope.
- The sample knowledge is written from the author's own earlier course projects, so it can be shared freely.
- Scanned PDFs are not OCR'd; they are rejected.
