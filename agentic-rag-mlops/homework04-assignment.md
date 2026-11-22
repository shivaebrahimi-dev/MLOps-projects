# MLOps Assignment: Agentic RAG Knowledge Assistant with CrewAI and Langfuse

Deadline: Midnight (11:59 PM), Monday, November 30, 2026.

Submission method: Send your homework by email.

Recipient: *(course instructor — address omitted from the public repository)*

Email subject: MLOps Homework #4

Attachment: Compress your submission into a ZIP file before sending.

Required content: Include your skills, implementation, and any supporting materials relevant to the homework.

## 1. Assignment overview

Build and test a small, production-minded Retrieval-Augmented Generation (RAG) application in Python. The application must let a user upload knowledge files, index them, and ask questions grounded in those files. The answering workflow must be implemented as a CrewAI agentic workflow and every agent/task execution must be traceable in Langfuse.

This is a **vibe-coding with specification** assignment: before writing implementation code, the student must use Spec Kit (or an equivalent specification-first workflow) to define the product requirements, technical design, acceptance criteria, and implementation tasks. The goal is not to build a large product. The goal is to demonstrate that the student can combine Python, RAG, agent orchestration, observability, testing, and evaluation practices.

## 2. Required technology combination

The solution must use all of the following:

1. **Python** for the application and tests.
2. **Spec Kit** (or the course-provided Spec Kit workflow) for requirements and implementation planning.
3. **CrewAI skill** supplied by the official page: install and configure it, then use CrewAI for the agentic answer workflow.
4. **Langfuse skill** supplied by the official page: install and configure it, then instrument the workflow and persist traces.
5. **One student-created skill**, named `rag-evaluation-report`, with the following frontmatter values:

   ```yaml
   user-invocable: true
   disable-model-invocation: true
   ```

   This skill (`rag-evaluation-report`) must be invoked explicitly by the user after the application has been implemented and tested. It must not run automatically during Agentic coding, upload or question answering.

## 3. User experience requirements

Implement a simple UI with at least two tabs or clearly separated views.

### Tab A: Knowledge upload and indexing

The user must be able to:

- Upload one or more supported files. At minimum support `.txt` and `.pdf`; `.md` is recommended.
- See the uploaded file names and basic processing status.
- Parse, chunk, embed, and store the documents in a retrievable knowledge base.
- Receive a clear success or error message.
- Avoid silently accepting an empty file or unsupported file type.

The application must preserve enough metadata to identify the source file and chunk when displaying citations.

### Tab B: Ask questions

The user must be able to:

- Enter a natural-language question.
- Receive an answer based on the indexed knowledge.
- See source citations or retrieved chunk references.
- See a safe response such as “I could not find this in the uploaded knowledge” when the evidence is insufficient.
- See a visible processing state while the agentic workflow is running.

The application should not claim unsupported facts. The system prompt and answer-generation logic must instruct the agent to use retrieved evidence and distinguish evidence from uncertainty. It would be mananged by a hallucination guardrail.

## 4. Minimum agentic workflow

Use CrewAI to implement a small, understandable workflow. A recommended design is:

1. **Retriever/Researcher agent**: receives the user question, queries the vector store, and returns the most relevant chunks with source metadata.
2. **Answer/Verifier agent**: uses only the retrieved context to draft an answer, identify missing evidence, and include citations.
3. **Crew/task orchestration**: defines the agents, tasks, inputs, expected outputs, and execution order. Sequential execution is sufficient; a more complex process is not required.

## 5. Langfuse observability requirements

Instrument the end-to-end question-answer workflow with Langfuse. At minimum, capture:

- A trace per user question.
- The user question and a non-sensitive request/session identifier.
- CrewAI agent and task spans or equivalent nested observations.
- Retrieval operation and retrieved document/chunk metadata.
- LLM generation input/output metadata, while avoiding secrets and unnecessary personal data.
- Latency and status for major steps.
- Token usage and model name when available.
- Errors and failed runs.

The README must explain how to configure Langfuse locally, how to confirm that traces are arriving, and how to run the application without exposing credentials in source control. A mock or disabled observability mode may be provided for tests, but the real Langfuse integration must be implemented and demonstrated with a trace screenshot or exported trace identifier.

## 6. Specification-first workflow (Spec kit)

Before implementation, create a `specs/` directory containing at least:

- `constitution.md`
- `spec.md`
- `plan.md`
- `tasks.md`

## 7. Testing requirements

Provide automated tests and a repeatable manual test procedure.

Minimum automated coverage:

- File type and empty-file validation.
- Document chunking and metadata preservation.
- Retrieval returns relevant evidence for at least one known question.
- Unknown or unsupported question produces an appropriately cautious answer or classification.
- The answer contains source references when evidence is available.
- Langfuse instrumentation can be disabled or mocked in unit tests without breaking the application.
- A health or smoke test exercises the main application path with a deterministic fake LLM or test double where practical.

Create a small evaluation dataset with at least five questions:

- Three answerable questions whose answers are present in the uploaded files.
- One unanswerable question.
- One question requiring information from more than one chunk or document.

For each question, record the expected answer points and expected source document(s). Do not use private or copyrighted material that the evaluator cannot access.

## 8. Student-created evaluation skill

Create the skill at `skills/rag-evaluation-report/SKILL.md` or in the .agents/skills directory. The skill must:

- Be named `rag-evaluation-report`.
- Declare `user-invocable: true`.
- Declare `disable-model-invocation: true`.
- State that it is invoked only when the user explicitly requests an evaluation report.
- Read the project specifications, test results, evaluation dataset, application configuration, and available Langfuse trace summaries.
- Produce a Markdown report, for example `reports/rag-evaluation-report.md`.
- Identify applicable metrics rather than inventing measured values.
- Include limitations, data-quality concerns, privacy considerations, and recommended next experiments.

The report should consider (Use your previous report -homework 03 as the basis for the evaluation metrics.), when applicable:

- **Retrieval**: Recall@k, Precision@k, Hit Rate@k, MRR, nDCG, context relevance, and chunk coverage.
- **Generation/RAG quality**: faithfulness/groundedness, answer relevance, context utilization, citation correctness, citation completeness, and unsupported-claim rate.
- **Task outcome**: correctness against expected answer points, abstention quality, and human preference or rubric score.
- **Agent workflow**: task success rate, tool/retrieval failure rate, number of retries, trace completeness, step latency, end-to-end latency, and cost/token usage.
- **Operations**: upload success rate, indexing latency, query throughput, error rate, and resource usage.
- **Safety/privacy**: prompt-injection detection rate, data leakage checks, secret exposure, and PII handling.

The skill must not fabricate Langfuse data. If it cannot access a metric, it must label it as unavailable and explain how to collect it.

## 9. Suggested repository structure

```text
agentic-rag-assignment/
├── README.md
├── pyproject.toml or requirements.txt
├── .env.example
├── app/
│   ├── ui.py
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── crew.py
│   ├── observability.py
│   └── config.py
├── data/
│   └── sample-knowledge/
├── evaluation/
│   ├── questions.jsonl
│   └── expected_answers.jsonl
├── specs/
│   ├── spec.md
│   ├── plan.md
│   ├── tasks.md
├── .agents/skills/
│   └── rag-evaluation-report/
│       └── SKILL.md
├── tests/
└── reports/
```

The exact structure may vary, but responsibilities must remain understandable and secrets must not be committed.

## 10. Required deliverables

Submit:

1. A runnable source repository.
2. The four specification files and evidence of one specification iteration.
3. Setup instructions for Python dependencies, CrewAI, Langfuse, the selected model provider, and the UI.
4. `.env.example` with variable names only.
5. Automated tests and test results.
6. The evaluation dataset and expected-answer rubric.
7. At least one Langfuse trace identifier or screenshot from a successful question run.
8. The user-invocable `rag-evaluation-report` skill.
9. A generated evaluation report produced by explicitly invoking that skill after testing.

## 11. Definition of done

The assignment is complete when a clean evaluator environment can install the documented dependencies, configure the required secrets, start the UI, upload sample files, ask a question, see a cited answer, and find a corresponding Langfuse trace. The evaluator can then explicitly invoke the evaluation skill and obtain a report that is honest about which metrics can be  measured and which require additional instrumentation.
