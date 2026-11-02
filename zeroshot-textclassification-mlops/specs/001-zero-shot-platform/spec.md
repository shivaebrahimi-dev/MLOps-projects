# Feature Specification: Zero-Shot Text Classification Platform

**Feature Branch**: `add-zeroshot-textclassification-spec`

**Created**: 2026-11-01

**Status**: Draft

**Input**: User description: "Build a small MLOps zero-shot text classification platform with an administrator page for downloading, caching, inspecting, and deleting models and managing candidate labels, and a separate user page for classifying text, with every classification result saved as JSON in MinIO." (Summarized from the homework file, section 12.)

## Purpose and Scope

The platform lets a user classify text into categories chosen at request time, without training a model, and keeps a structured record of every result for later analysis. Its wider purpose is to practice spec-driven development and basic MLOps: reproducibility, model lifecycle management, testing, logging, and clear separation of responsibilities.

**In scope**: model management, candidate-label management, text classification, prediction storage, and the supporting practices in the Non-Functional Requirements.

**Out of scope** (per the constitution): admin authentication, Prometheus metrics, storage lifecycle policies, a prediction-history page, CSV export, batch classification, and model comparison.

**Named technologies**: The constitution and assignment mandate Hugging Face, MinIO, and Docker Compose as product constraints, so requirements name them where needed. All other implementation choices belong in the plan.

## Users and Roles

- **Administrator**: Manages models and candidate labels. Works in a local or trusted environment, without authentication in this version.
- **User**: Submits text and reads results. Cannot change models or label sets.
- **Developer**: Sets up, tests, and maintains the platform using only the repository and its documentation.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Classify text and keep a record (Priority: P1)

A user enters text, selects a downloaded model and a label set, and submits it. The user sees the predicted class, confidence, ranked labels, model used, and timestamp, and the result is saved for later analysis.

**Why this priority**: This is the platform's core value and the behavior the constitution makes non-negotiable.

**Independent Test**: With one model already available and one label set configured, submit a sample text, confirm the displayed result, and find the matching JSON object in the bucket.

**Acceptance Scenarios**:

1. **Given** a ready model and a valid label set, **When** the user submits "The team won the championship after a very close game." with labels including `sports`, **Then** the page shows `sports` as the predicted class, its confidence score, all labels ranked by score, the model used, and the timestamp.
2. **Given** a successful classification, **When** the user opens the bucket, **Then** a JSON object exists at `predictions/YYYY/MM/DD/{request_id}.json` containing all required fields.
3. **Given** the bucket does not exist, **When** the first result is saved, **Then** the bucket is created automatically and the result is stored.
4. **Given** empty or whitespace-only text, **When** the user submits it, **Then** the request is rejected with a clear message and nothing is stored.
5. **Given** no ready model or no valid label set, **When** the user opens the page or submits, **Then** a clear message explains what is missing and how to resolve it.

---

### User Story 2 - Download and inspect models (Priority: P2)

An administrator views the configured models, selects one or more, downloads them, and sees each model's status, identifier, revision, and storage location.

**Why this priority**: Classification needs a ready model, and this is how one becomes available. It ranks below Story 1 because Story 1 delivers the end-user value and can be tested with a pre-provisioned model.

**Independent Test**: With an empty model directory, download a configured model, then request it again and confirm that nothing is downloaded the second time.

**Acceptance Scenarios**:

1. **Given** the default configuration, **When** the administrator opens the admin page, **Then** `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` and `facebook/bart-large-mnli` are both listed with their status.
2. **Given** a missing model, **When** the administrator downloads it, **Then** it is stored in the controlled directory, marked ready, and offered on the classification page.
3. **Given** a ready model, **When** the administrator requests it again, **Then** no download occurs and the page reports that it is already available.
4. **Given** a download fails or is interrupted, **When** the administrator refreshes the list, **Then** the model is not ready, no invalid model directory remains, and a clear error was shown.
5. **Given** a new model entry is added to the configuration, **When** the list is refreshed, **Then** it appears without application code changes.

---

### User Story 3 - Manage candidate labels (Priority: P3)

An administrator adds and removes labels, saves named label sets, and makes them available for classification.

**Why this priority**: Users need label sets, but a default set lets Story 1 work before any are customized.

**Independent Test**: Create and save a label set, restart the application, and confirm that the set is still available and selectable.

**Acceptance Scenarios**:

1. **Given** a label set, **When** the administrator adds or removes a label and saves, **Then** the change persists across application restarts.
2. **Given** fewer than two labels, **When** the administrator saves, **Then** saving is refused with a clear message.
3. **Given** an empty, whitespace-only, or duplicate label, **When** the administrator adds it, **Then** it is rejected with a clear message.

---

### User Story 4 - Delete models and clear the cache (Priority: P4)

An administrator deletes a selected local model or clears the Hugging Face cache to reclaim disk space.

**Why this priority**: This completes the model lifecycle, but the core workflow runs without it.

**Independent Test**: Delete a ready model and confirm that its status changes and it is no longer offered for classification. Then clear the cache when the cache directory is missing and confirm that no failure occurs.

**Acceptance Scenarios**:

1. **Given** a ready model, **When** the administrator explicitly confirms deletion, **Then** its local files are removed, its status becomes not downloaded, and it is no longer selectable.
2. **Given** the delete action is started but not confirmed, **When** the administrator cancels, **Then** nothing is removed.
3. **Given** a cache exists, **When** the administrator explicitly confirms cache cleanup, **Then** the cache is cleared and the outcome is reported.
4. **Given** the cache directory does not exist, **When** the administrator clears the cache, **Then** the action completes with an informative message.

---

### User Story 5 - Reproduce and verify the project (Priority: P5)

A developer new to the project follows the repository's documentation to start the services, run the full workflow, and run the tests.

**Why this priority**: The assignment's definition of done depends on reproducibility, but it builds on the working features above.

**Independent Test**: On a machine with only the documented prerequisites, follow the README from a fresh clone through a stored prediction and a passing test run.

**Acceptance Scenarios**:

1. **Given** a fresh clone, **When** the developer follows the quick start, **Then** storage and the application start using the documented commands.
2. **Given** storage is running, **When** the developer runs the documented test commands, **Then** the unit and integration tests pass.
3. **Given** a change is proposed, **When** CI runs, **Then** quality checks and automated tests report a clear pass or fail.

---

### Edge Cases

- **Storage unavailable during classification**: The user sees the result and a clear error that it was not saved, with the request ID. The failure is logged and no local file replaces storage.
- **Storage unavailable at startup**: The application still opens, and the problem is reported clearly when storage is needed.
- **Network or Hugging Face failure during a download**: The model stays not ready, partial files are removed, and credentials never appear in the message.
- **Gated or nonexistent model identifier**: A clear error is shown, and other models are unaffected.
- **Second download of a model already downloading**: The second request is refused with a clear message.
- **Selected model was deleted or corrupted before submission**: The request is rejected with a message directing the user to choose an available model.
- **Malformed model or label configuration**: A clear error names the file and the problem, without exposing secrets.
- **Unsafe model identifiers**: Identifiers that could write outside the controlled directory, such as those containing `..`, are rejected.
- **Very long input text**: Text above the documented limit is rejected with a message stating the limit.
- **Labels that differ only by case or surrounding spaces**: They are treated as duplicates.
- **Insufficient memory or disk space**: The operation fails with a clear message, and the application stays available.

## Requirements *(mandatory)*

### Functional Requirements

**Model management**

- **FR-001**: The system MUST list every model defined in configuration, with its name, identifier, status, and revision when available.
- **FR-002**: The system MUST support at least two zero-shot models and MUST include `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` and `facebook/bart-large-mnli` by default.
- **FR-003**: Administrators MUST be able to select one or more models and download them from Hugging Face.
- **FR-004**: The system MUST display each model's download and availability status, the local model directory, and the cache location.
- **FR-005**: Administrators MUST be able to refresh the list of locally available models.
- **FR-006**: Adding a supported model MUST require only a configuration change.

**Caching and deletion**

- **FR-007**: The system MUST check for a complete local copy before downloading and MUST NOT download a model that is already ready.
- **FR-008**: The system MUST store models in a configurable, controlled directory, using paths that cannot resolve outside it.
- **FR-009**: The system MUST mark a model ready only after its download is complete and verified, so that partial downloads are never treated as ready.
- **FR-010**: A failed or interrupted download MUST NOT leave an invalid model directory behind.
- **FR-011**: Administrators MUST be able to delete a selected local model, only after an explicit confirmation.
- **FR-012**: Administrators MUST be able to clear the Hugging Face cache, only after an explicit confirmation, and this MUST NOT remove ready models from the controlled directory.
- **FR-013**: Model and cache operations MUST display meaningful success or error messages.

**Candidate labels**

- **FR-014**: Administrators MUST be able to add labels, remove labels, and save named label sets.
- **FR-015**: The system MUST reject empty labels and duplicate labels, and MUST require at least two labels in a set.
- **FR-016**: Label sets MUST persist across restarts in a documented configuration file, and a default set MUST be available on first run.

**Classification**

- **FR-017**: Users MUST be able to enter or paste text, select a ready model, select a label set, and submit the text for classification.
- **FR-018**: The system MUST display the input text, predicted class, confidence score, all labels ranked by descending score, the model used, and the prediction timestamp.
- **FR-019**: The predicted class MUST be the highest-scoring label.
- **FR-020**: The system MUST remove control characters from input, reject empty text and text above the documented maximum length, and MUST treat input text as plain text rather than as executable or formatted content.
- **FR-021**: The system MUST show a clear message when no ready model or valid label set is available.
- **FR-022**: The administrator and classification workflows MUST be separate pages.

**Prediction storage**

- **FR-023**: The system MUST save every successful classification as one JSON object in the configured MinIO bucket.
- **FR-024**: The default bucket MUST be `zeroshot-learning-output`, configurable through `MINIO_BUCKET`, and the system MUST create the bucket automatically if it does not exist.
- **FR-025**: Objects MUST be stored at `predictions/YYYY/MM/DD/{request_id}.json`, with the date taken from the UTC prediction timestamp.
- **FR-026**: Each object MUST contain the request ID, timestamp, input text, predicted class, confidence, candidate labels, ranked labels and scores, model name, Hugging Face model identifier, model revision when available, and application version. It MAY contain a user or session identifier and inference latency.
- **FR-027**: Each request MUST have a unique request ID.
- **FR-028**: If saving fails, the system MUST show a clear error stating that the result was not saved, and MUST NOT treat local storage as a replacement.

### Non-Functional Requirements

- **NFR-001 (Reproducibility)**: The Python, PyTorch, Transformers, and other dependency versions, model identifiers, and storage configuration MUST be pinned or documented.
- **NFR-002 (Configuration)**: Credentials and environment-specific settings MUST come from environment variables, with placeholder values only in `.env.example`.
- **NFR-003 (Observability)**: Model downloads, inference, cache cleanup, and storage writes MUST produce structured logs with useful metadata.
- **NFR-004 (Maintainability)**: User interface, model management, inference, configuration, and storage responsibilities MUST remain separate.
- **NFR-005 (Resource use)**: The system MUST NOT load multiple large models into memory unnecessarily and SHOULD reuse initialized models.
- **NFR-006 (Testing: model management)**: Automated tests MUST cover configuration loading, missing-model detection, avoiding repeat downloads, safe paths, deletion, cleanup with a missing cache, and failed downloads.
- **NFR-007 (Testing: classification)**: Automated tests MUST cover valid classification, label validation, empty text, fewer than two labels, ranking, and the predicted class being the highest-scoring label.
- **NFR-008 (Testing: storage)**: Automated tests MUST cover bucket creation, valid JSON, the date-based path, required fields, and clear storage errors. At least one end-to-end test MUST verify a write to a running MinIO.
- **NFR-009 (Test weight)**: Tests that would need large model downloads MUST use mocks or lightweight fixtures whenever possible.
- **NFR-010 (Delivery)**: MinIO MUST start locally through Docker Compose, and CI MUST run quality checks and automated tests.
- **NFR-011 (Documentation)**: The README MUST cover everything the assignment's README notice lists, for a developer new to the project.

### Security Considerations

- **SEC-001**: Secrets, `.env` files, model files, and cache directories MUST NOT be committed to Git.
- **SEC-002**: Logs and error messages MUST NOT contain credentials or tokens. Logs MUST NOT contain input text unless it is needed to diagnose a failure.
- **SEC-003**: Because administrator authentication is out of scope, the README MUST state that the admin page is intended only for local development or trusted environments.
- **SEC-004**: Stored predictions contain full input text, so the README MUST warn users about storing sensitive text.
- **SEC-005**: The default MinIO credentials are for local development only, and the README MUST say so.

### Key Entities

- **Model**: A configured zero-shot model. Attributes: display name, identifier, task, status (not downloaded, downloading, ready, or failed), revision, and local location.
- **Label Set**: A named, persisted group of at least two unique candidate labels.
- **Classification Request**: One submission of text with a selected model and label set, identified by a unique request ID.
- **Prediction Record**: The stored result of a successful request, holding the fields in FR-026. Each request has at most one record.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: An administrator can see at least two configured models, download each one, and select it for classification.
- **SC-002**: Requesting a ready model again transfers no model files and reports its status within 5 seconds.
- **SC-003**: 100% of simulated failed downloads leave the model not ready and leave no invalid model directory.
- **SC-004**: After deletion, a model is unavailable for classification, and clearing the cache reports its outcome, including when no cache exists.
- **SC-005**: A user can go from entering text to viewing ranked results in under 1 minute once a model is ready, on the documented minimum hardware.
- **SC-006**: 100% of successful classifications in the end-to-end test produce one stored record containing every required field, at the required path.
- **SC-007**: 100% of tested invalid inputs and tested failures of model availability or storage produce a clear message, with no crash and no exposed secret.
- **SC-008**: A new developer can reach a first stored prediction by following only the README, in under 30 minutes excluding model download time.
- **SC-009**: All unit tests pass, and integration tests pass when storage is running.
- **SC-010**: The repository's tracked files contain no secrets, model files, or cache directories.
- **SC-011**: The expected demonstration in the assignment's section 13 can be completed from start to finish in one session.
- **SC-012**: The constitution, specification, plan, data model, and task breakdown are all present in the repository.

## Assumptions

- **Classification mode**: Each request uses single-label classification, so scores are compared across labels and the top score is the predicted class.
- **Input limit**: The default maximum input length is 10,000 characters, and it is configurable.
- **Storage failure behavior**: When saving fails, the user still sees the result along with a clear "not saved" error. The assignment requires a clear message but doesn't say whether to show the result.
- **Model revision**: The system records the exact revision resolved at download time when available. Otherwise it records that the revision is unavailable.
- **Timestamps**: Timestamps are in UTC and ISO 8601 format.
- **Cache separation**: Ready models in the controlled directory are separate from the Hugging Face cache, so clearing the cache does not remove them.
- **Default labels**: The default label set is `sports`, `politics`, `technology`, `finance`, `health`, and `education`.
- **Single operator**: The platform is used by one administrator in a local or trusted environment, so simultaneous administrators are not a design goal.
- **Hardware**: Classification must work on a CPU. A GPU is optional, and minimum hardware is documented in the README.
- **Prerequisites**: Developers have Python and Docker available, and internet access to download models.
- **Bucket name**: `zeroshot-learning-output` replaces the original assignment's underscore name, as recorded in constitution decision DR-001.
