---

description: "Task list for the Zero-Shot Text Classification Platform"
---

# Tasks: Zero-Shot Text Classification Platform

**Input**: Design documents from `specs/001-zero-shot-platform/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Included because the specification requires them (NFR-006 to NFR-009). The constitution does not require
test-first development, so each test task may be completed alongside its implementation task.

**Organization**: Tasks are grouped by user story so each story can be implemented and tested independently.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1 to US5)
- All paths are relative to `zeroshot-textclassification-mlops/`, unless marked as the repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Protect the public repository, confirm MinIO works, and initialize the project

- [x] T001 Create `.gitignore` covering `.env`, `.venv/`, `.cache/`, `models/*` except `models/.gitkeep`, and Python, pytest, and ruff caches, before any model or secret exists
- [x] T002 Create `docker-compose.yml` with a `minio` service using `${MINIO_IMAGE:-bitnamilegacy/minio:2025.7.23-debian-12-r5}`, a `minio-console` service using `${MINIO_CONSOLE_IMAGE:-bitnamilegacy/minio-object-browser:2.0.2-debian-12-r4}`, ports 9000 and 9001, credentials from the environment, and a named data volume
- [x] T003 Start MinIO, confirm its data path, console, and survival of a restart, and confirm whether `zeroshot_learning_output` is rejected and `zeroshot-learning-output` accepted. Record the results in `specs/001-zero-shot-platform/research.md` (R2, R3) and update DR-001 and DR-003 in `.specify/memory/constitution.md`. If the image fails, switch to `pgsty/minio:RELEASE.2026-08-04T00-00-00Z` and record that instead
- [x] T004 [P] Create `requirements.txt` with the pinned versions from research R1: torch 2.15.0, transformers 5.22.1, huggingface_hub 2.4.0, streamlit 1.66.1, minio 7.2.20, PyYAML 6.0.3, python-dotenv 1.2.4, pytest 9.2.0, and ruff 0.17.3
- [x] T005 [P] Create `pyproject.toml` with project metadata, Python 3.12, ruff settings, pytest paths, and the `integration` marker
- [x] T006 [P] Create `.env.example` with every variable in `specs/001-zero-shot-platform/contracts/config-files.md`, using placeholders and noting that `minioadmin` is for local use only
- [x] T007 [P] Create `configs/models.yaml` with both models and their pinned revisions, and `configs/labels.yaml` with the `general-topics` default set, per `specs/001-zero-shot-platform/contracts/config-files.md`
- [x] T008 [P] Create the `app/`, `app/pages/`, `app/services/`, `app/utils/`, `tests/unit/`, and `tests/integration/` packages, plus `models/.gitkeep`
- [x] T009 [P] Create `Makefile` targets for install, MinIO start and stop, app start, lint, unit tests, and integration tests

**Checkpoint**: MinIO runs in Docker Desktop, and no secrets or models can be committed by accident

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared code that every user story needs

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T010 [P] Implement typed errors with safe user-facing messages in `app/utils/errors.py`
- [x] T011 [P] Implement structured JSON logging in `app/utils/logging_config.py`, excluding credentials and tokens, with a default that omits input text
- [x] T012 Implement settings in `app/utils/config.py` that load `.env` once before Hugging Face libraries are imported, apply the defaults in the configuration contract, name missing required variables, and keep secrets out of their text representation
- [x] T013 Implement model configuration loading and read-only status in `app/services/model_manager.py`, enforcing "Required and unique. Hugging Face format `org/name`, with no `..`, absolute paths, or backslashes", reporting `not_downloaded`, `downloading`, `ready`, or `failed`, and requiring "both the final folder and a valid completion marker" for `ready`
- [x] T014 Implement label-set loading and validation in `app/services/label_manager.py`: names "unique after trimming, ignoring case", and labels "At least 2 labels. Each is trimmed and non-empty. No case-insensitive duplicates."
- [x] T015 Implement shared service instances with `st.cache_resource` in `app/ui_state.py`
- [x] T016 Create the home page and navigation in `app/streamlit_app.py`
- [x] T017 [P] Add unit tests for settings, safe errors, and log redaction in `tests/unit/test_config.py`
- [x] T018 [P] Add unit tests for model configuration loading, missing-model detection, safe paths, and readiness rules in `tests/unit/test_model_manager.py`
- [x] T019 [P] Add unit tests for label validation, including case-only duplicates, in `tests/unit/test_label_manager.py`
- [x] T020 [P] Add an architecture test in `tests/unit/test_architecture.py` proving that nothing in `app/services/` or `app/utils/` imports Streamlit

**Checkpoint**: Foundation ready, and the app opens with an empty home page

---

## Phase 3: User Story 1 - Classify text and keep a record (Priority: P1) 🎯 MVP

**Goal**: A user classifies text, sees ranked results, and the result is saved to MinIO

**Independent Test**: With a ready model and a label set, submit sample text, confirm the displayed result, and find the matching JSON object in the bucket

### Implementation for User Story 1

- [x] T021 [US1] Implement `app/services/classifier.py`: load only `ready` models from local paths, keep at most one pipeline in memory, run single-label zero-shot inference, raise an error unless the model is a `torch.nn.Module`, measure latency, and reject text unless it is "Not empty after trimming, and at most `MAX_INPUT_CHARS` characters (default 10,000)"
- [x] T022 [US1] Implement `app/services/minio_storage.py`: build a client with short timeouts and limited retries, create the bucket when needed while tolerating an existing one, build records with UUID4 request IDs and UTC ISO 8601 timestamps, and write `application/json` objects to `predictions/YYYY/MM/DD/{request_id}.json`
- [x] T023 [US1] Implement `app/pages/2_Classify_Text.py` per `specs/001-zero-shot-platform/contracts/ui-pages.md`, showing the predicted class, confidence, ranked labels, model, revision, timestamp, request ID, and storage outcome, and displaying input as plain text
- [x] T024 [US1] Add the failure paths to `app/pages/2_Classify_Text.py`: empty or long text, no ready model, invalid label set, and a storage failure that shows the result with a clear "not saved" error and the request ID
- [x] T025 [US1] Log inference and storage writes with request ID, model, duration, and outcome in `app/services/classifier.py` and `app/services/minio_storage.py`

### Tests for User Story 1

- [x] T026 [P] [US1] Add unit tests in `tests/unit/test_classifier.py` with a fake pipeline: valid classification, empty text, length limit, fewer than two labels, descending ranking, predicted class equal to the top label, and rejection of a non-PyTorch model
- [x] T027 [P] [US1] Add unit tests in `tests/unit/test_minio_storage.py` with a fake client: bucket creation, valid JSON, required fields against `specs/001-zero-shot-platform/contracts/prediction-record.schema.json`, the date-based path, and clear storage errors
- [x] T028 [P] [US1] Add integration tests in `tests/integration/test_minio_storage.py` against running MinIO for automatic bucket creation, stored JSON, the object path, and connection failure
- [x] T029 [US1] Add an end-to-end test in `tests/integration/test_end_to_end.py` that runs a tiny test model through the classifier and confirms the stored record in MinIO
- [x] T030 [P] [US1] Add a page smoke test with Streamlit's `AppTest` and fake services in `tests/unit/test_classify_page.py`

**Checkpoint**: User Story 1 works with a pre-provisioned model, which is the MVP

---

## Phase 4: User Story 2 - Download and inspect models (Priority: P2)

**Goal**: An administrator downloads configured models and sees their status

**Independent Test**: With an empty model directory, download a model, then request it again and confirm that nothing downloads the second time

### Implementation for User Story 2

- [x] T031 [US2] Add downloads to `app/services/model_manager.py` per the plan: refuse a second download through the lock file, fetch the pinned revision into `MODEL_DIR/.tmp/` with only configuration, tokenizer, and `model.safetensors` files, write `.download-complete.json`, rename into place, and on failure clean up and record a secret-free error
- [x] T032 [US2] Add the model table, locations, download, and refresh areas to `app/pages/1_Admin.py` per `specs/001-zero-shot-platform/contracts/ui-pages.md`, with the local-use notice and per-model success or error messages
- [x] T033 [US2] Log download start, completion, skips, and failures with model ID, revision, and duration in `app/services/model_manager.py`

### Tests for User Story 2

- [x] T034 [P] [US2] Add unit tests in `tests/unit/test_model_download.py` with a fake downloader: existing models are not downloaded again, failed downloads leave no final folder or temporary files, stale locks are removed, a second download is refused, and a newly configured model appears without code changes
- [x] T035 [P] [US2] Add a page smoke test with fake services in `tests/unit/test_admin_page.py`

**Checkpoint**: Stories 1 and 2 together give the full download-to-prediction workflow with real models

---

## Phase 5: User Story 3 - Manage candidate labels (Priority: P3)

**Goal**: An administrator maintains label sets that persist and can be selected for classification

**Independent Test**: Create and save a label set, restart the app, and confirm that it is still selectable

### Implementation for User Story 3

- [x] T036 [US3] Add adding, removing, and saving to `app/services/label_manager.py`, using a temporary file and replacement so a failed save cannot corrupt `configs/labels.yaml`
- [x] T037 [US3] Add the label area to `app/pages/1_Admin.py`, showing validation errors beside the form

### Tests for User Story 3

- [x] T038 [P] [US3] Add unit tests in `tests/unit/test_label_persistence.py` for saved sets surviving a reload, refused invalid saves, and an unchanged file after a failed save

**Checkpoint**: Label changes on the admin page appear on the classification page

---

## Phase 6: User Story 4 - Delete models and clear the cache (Priority: P4)

**Goal**: An administrator reclaims disk space through explicit actions

**Independent Test**: Delete a ready model and confirm it is no longer offered, then clear a missing cache without failure

### Implementation for User Story 4

- [x] T039 [US4] Add model deletion to `app/services/model_manager.py`, unloading the model through `app/services/classifier.py` if it is active, and confirming the path is inside `MODEL_DIR` before removing it
- [x] T040 [US4] Add cache cleanup to `app/services/model_manager.py`, removing only `hub` and `xet` under `HF_HOME`, preserving saved tokens, reporting what was removed, and handling a missing cache
- [x] T041 [US4] Add the delete and cache areas to `app/pages/1_Admin.py`, with buttons disabled until a confirmation checkbox is ticked, and log both actions

### Tests for User Story 4

- [x] T042 [P] [US4] Add unit tests in `tests/unit/test_model_cleanup.py` for deletion, path safety, a missing cache, preserved tokens, and ready models surviving cache cleanup

**Checkpoint**: The full model lifecycle works

---

## Phase 7: User Story 5 - Reproduce and verify the project (Priority: P5)

**Goal**: A developer new to the project can set up, run, and verify it from the repository alone

**Independent Test**: From a fresh clone, follow the README through a stored prediction and a passing test run

### Implementation for User Story 5

- [x] T043 [P] [US5] Create `Dockerfile` for the app with Python 3.12 and pinned dependencies, and add it to `docker-compose.yml` under an opt-in `app` profile with `MINIO_ENDPOINT=minio:9000`
- [x] T044 [US5] Create `.github/workflows/zeroshot-ci.yml` at the repository root, filtered to `zeroshot-textclassification-mlops/**`, running from that folder with CPU-only PyTorch, a MinIO service container, ruff, unit tests, and integration tests that fail instead of skipping
- [x] T045 [US5] Write `README.md` covering every item in the assignment's README notice: purpose, architecture, code structure, module responsibilities, models, model and cache workflow, labels, bucket and output format, prerequisites, environment variables, quick start, MinIO and Streamlit startup, downloading a model, classifying, verifying in MinIO, tests, troubleshooting, and security limitations
- [x] T046 [US5] Disclose the project's departures from the assignment in `README.md`: the bucket name (DR-001), MinIO image (DR-003), `TRANSFORMERS_CACHE`, CI location, `spec.md` file name, and use of Claude Code instead of GitHub Copilot, and explain why Clear Cache may find little to remove

**Checkpoint**: All user stories work, and CI passes on GitHub

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Final checks before submission

- [x] T047 Run ruff and the full test suite, and fix any failures across `app/` and `tests/`
- [x] T048 Run every scenario in `specs/001-zero-shot-platform/quickstart.md` with both real models, measure SC-002, SC-005, and SC-008, and record the results in `specs/001-zero-shot-platform/research.md`
- [ ] T049 Rehearse the assignment's section 13 demonstration from start to finish, using `README.md` only, and fix any step that fails
- [x] T050 Check `README.md` against the final code and the assignment's section 11 acceptance criteria, and run `/speckit-analyze` to check that the Spec Kit documents agree
- [x] T051 Remove the Sync Impact Report comment from `.specify/memory/constitution.md`
- [x] T052 Before the final push, confirm that Git tracks no secrets, `.env` files, model files, or cache directories, and that no tracked file contains personal names or email addresses of course staff

**Checkpoint**: Ready to merge to `main` and submit by email before November 14, 2026

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies. T001 comes first, and T003 needs T002 and a running Docker Desktop.
- **Foundational (Phase 2)**: Depends on Setup, and blocks all user stories.
- **User Stories (Phases 3 to 7)**: Depend on Foundational, and proceed in priority order.
- **Polish (Phase 8)**: Depends on all user stories.

### User Story Dependencies

- **US1 (P1)**: No dependency on other stories. It uses a tiny test model until US2 exists.
- **US2 (P2)**: No dependency on other stories. Together with US1, it enables real-model classification.
- **US3 (P3)**: No dependency on other stories, because Foundational provides the default label set.
- **US4 (P4)**: Needs downloaded models from US2 for real use, and fixtures for its tests. T039 also uses the classifier from US1.
- **US5 (P5)**: T045 and T046 document the finished behavior, so they follow US1 to US4. T043 and T044 can start earlier.

### Within Each User Story

- Services come before pages.
- Tests may be written with or right after the code they cover.
- A story's tests must pass before it is considered complete.

### Parallel Opportunities

Tasks marked [P] touch different files. Tasks that change `app/pages/1_Admin.py` or `app/services/model_manager.py`
(T013, T031 to T033, T037, T039 to T041) must run one at a time.

---

## Parallel Example: User Story 1

```text
Task: "Add unit tests in tests/unit/test_classifier.py"
Task: "Add unit tests in tests/unit/test_minio_storage.py"
Task: "Add integration tests in tests/integration/test_minio_storage.py"
Task: "Add a page smoke test in tests/unit/test_classify_page.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phases 1 and 2.
2. Complete Phase 3.
3. Stop and confirm that a prediction reaches MinIO.

### Incremental Delivery

Add US2, US3, and US4 in turn, testing each one. Then add US5 and finish with the final checks. Commit after each
task or logical group, and leave enough time before the deadline for T048 and T049, because they use real model
downloads.

## Assignment Coverage

| Assignment section 6.4 example | Tasks |
|---|---|
| 1. Initialize the Python project | T004, T005, T008 |
| 2. Create the model configuration | T007, T013 |
| 3. Create the labels configuration | T007, T014 |
| 4. Implement environment-based configuration | T006, T012 |
| 5. Implement model download | T031 |
| 6. Implement model cache detection | T013, T031 |
| 7. Implement model deletion | T039 |
| 8. Implement Hugging Face cache cleanup | T040 |
| 9. Implement zero-shot inference | T021 |
| 10. Create the MinIO client | T022 |
| 11. Create the MinIO bucket automatically | T022 |
| 12. Implement prediction persistence | T022 |
| 13. Build the Streamlit admin page | T032, T037, T041 |
| 14. Build the Streamlit classification page | T023, T024 |
| 15. Add input validation and error handling | T010, T014, T021, T024 |
| 16. Add unit tests | T017 to T020, T026, T027, T030, T034, T035, T038, T042 |
| 17. Add MinIO integration tests | T028, T029 |
| 18. Add Docker Compose support | T002, T043 |
| 19. Add CI validation | T044 |
| 20. Write setup and troubleshooting documentation | T045, T046 |
