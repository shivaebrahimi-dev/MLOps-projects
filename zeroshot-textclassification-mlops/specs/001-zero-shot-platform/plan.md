# Implementation Plan: Zero-Shot Text Classification Platform

**Branch**: `add-zeroshot-textclassification-spec` | **Date**: 2026-11-01 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-zero-shot-platform/spec.md`

## Summary

Build a local Streamlit application with separate admin and classification pages. Thin pages call independent
services for model management, labels, inference, and storage. Models download from Hugging Face at pinned
revisions into a controlled directory, using an atomic process so partial downloads never count as ready. Inference
uses the Transformers zero-shot pipeline on PyTorch, and each successful result is saved as JSON in MinIO, which runs
locally through Docker Compose. Decisions and evidence are in [research.md](research.md).

## Technical Context

**Language/Version**: Python 3.12

**Primary Dependencies**: PyTorch 2.15.0, Transformers 5.22.1, huggingface_hub 2.4.0, Streamlit 1.66.1, minio 7.2.20, PyYAML 6.0.3, python-dotenv 1.2.4

**Storage**: MinIO for prediction records; local files for models and YAML configuration

**Testing**: pytest 9.2.0, with mocks for unit tests and a running MinIO for integration tests; ruff 0.17.3 for linting and formatting

**Target Platform**: Local development on Windows, plus Linux in CI and the optional container

**Project Type**: Single-project web application

**Performance Goals**: The spec's SC-002 and SC-005 targets, which are still to be measured

**Constraints**: Must work on CPU, keep at most one pipeline loaded at a time, and pin versions throughout

**Scale/Scope**: One administrator and light local use, with 2 models configured by default

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Plan | Result |
|---|---|---|
| I. Purpose and Scope | Includes all required workflows, and no out-of-scope features | Pass |
| II. Separation of Concerns | Pages only call services, and a test verifies that services don't import Streamlit | Pass |
| III. Reproducibility | Pinned dependencies, model revisions, and MinIO image | Pass |
| IV. Configuration and Secrets | Environment variables and YAML, with models added by configuration only | Pass |
| V. Model Lifecycle | Atomic downloads, completion marker, confirmed deletion and cleanup, single loaded pipeline | Pass |
| VI. Data Persistence | MinIO is the only prediction store, with a configurable bucket that is created automatically | Pass, with DR-001 |
| VII. Testing | Unit, integration, and end-to-end tests, with mocks for large models | Pass |
| VIII. Error Handling | Typed service errors mapped to the messages in [ui-pages.md](contracts/ui-pages.md) | Pass |
| IX. Observability | Structured JSON logs for all required operations | Pass |
| X. Security and Privacy | Secret-free logs, `.gitignore`, and a README notice on admin access | Pass |
| XI. Documentation | README written against the final implementation | Pass |
| Technology Standards | Core stack only, with supporting libraries in DR-002 and the image source in DR-003 | Pass, with DR-002 and DR-003 |
| Quality Gates | ruff and pytest run in CI | Pass |
| Spec Kit Workflow | Plan and data model follow the reviewed constitution and specification | Pass |

**Post-design re-check**: Pass. The design added `python-dotenv` and a replacement MinIO image, so the constitution
was amended to v1.0.1 before this plan was finished. No violations need justifying.

## Architecture

```text
Streamlit pages (UI only)
  ├── Admin page ───────────┐
  └── Classify page ────────┤
                            ▼
                  app/ui_state.py (shared service instances)
                            │
      ┌─────────────┬───────┴─────┬────────────────┐
      ▼             ▼             ▼                ▼
 ModelManager  LabelManager   Classifier     PredictionStorage
      │             │             │                │
 Hugging Face  labels.yaml   PyTorch model     MinIO bucket
 models/ dir                 from models/ dir

 Shared by services: config.py, errors.py, logging_config.py
```

**Classification flow**: The page validates selections through the services, `Classifier` returns a result,
`PredictionStorage` saves the record, and the page displays both outcomes. A storage failure doesn't discard the
result (spec FR-028).

## Main Components

| Component | File | Responsibility |
|---|---|---|
| Home page | `app/streamlit_app.py` | Project overview and navigation |
| Admin page | `app/pages/1_Admin.py` | Model, cache, and label management interface |
| Classification page | `app/pages/2_Classify_Text.py` | Text input and result display |
| Service access | `app/ui_state.py` | Creates shared services with `st.cache_resource` |
| Model manager | `app/services/model_manager.py` | Configuration, status, download, deletion, and cache cleanup |
| Label manager | `app/services/label_manager.py` | Label validation and persistence |
| Classifier | `app/services/classifier.py` | Pipeline lifecycle, input validation, and inference |
| Storage | `app/services/minio_storage.py` | Bucket creation, record building, and writes |
| Configuration | `app/utils/config.py` | Loads and validates environment settings |
| Errors | `app/utils/errors.py` | Typed errors with safe user messages |
| Logging | `app/utils/logging_config.py` | Structured JSON logging |

## Technology Choices

The constitution sets the core stack. Other choices are explained in [research.md](research.md):
`huggingface_hub` for downloads (R4), `minio` for storage, ruff as one tool for lint and format checks (R1), and a
pinned Bitnami MinIO image (R2).

## Streamlit Page Design

The admin and classification pages follow [ui-pages.md](contracts/ui-pages.md). Downloads and inference run
synchronously with visible progress, which suits one local operator and avoids background-job complexity.

## Model Download and Caching

1. Validate the identifier and confirm that its path stays within `MODEL_DIR`.
2. If the model is ready, report that and stop.
3. Take the per-model lock, refusing if another download holds it.
4. Download the pinned revision into `MODEL_DIR/.tmp/`, with only the needed files.
5. Check that the configuration, tokenizer, and weights are present, then write the completion marker.
6. Rename the folder to `MODEL_DIR/{org}/{name}`, and release the lock.
7. On failure, remove temporary files, record the safe error, and release the lock.

**Deletion**: Requires confirmation in the UI, unloads the model if it's active, then removes its folder.

**Cache cleanup**: Requires confirmation in the UI, then removes only `hub` and `xet` under `HF_HOME` (R5).

## Hugging Face Integration

`snapshot_download` fetches models, reading `HF_TOKEN` when one is set. Inference loads only from the local ready
folder using `pipeline("zero-shot-classification", ...)` in single-label mode, then confirms the model is a PyTorch
module (R7). The revision recorded with each prediction comes from the completion marker.

## MinIO Integration

The client is built from environment settings with short timeouts and limited retries, so an unavailable server
fails quickly with a clear message. The default client retries could otherwise leave the page waiting for a long time.
The bucket is created when first needed and tolerates already existing. Records follow
[prediction-record.schema.json](contracts/prediction-record.schema.json) and are stored as `application/json`.

## Configuration Strategy

Settings and files are defined in [config-files.md](contracts/config-files.md). `.env` is loaded once at startup,
before Hugging Face libraries are imported, so that `HF_HOME` takes effect. Settings are validated when the application
starts, and secrets are excluded from their text representation.

## Testing Strategy

| Level | Location | Covers |
|---|---|---|
| Unit | `tests/unit/` | Spec NFR-006 to NFR-008, config, labels, and architecture rules, using fakes and mocks |
| Integration | `tests/integration/` | Bucket creation, stored JSON, paths, and connection failures, against running MinIO |
| End-to-end | `tests/integration/test_end_to_end.py` | Classifier with a tiny test model through to a stored record |
| Page smoke | `tests/unit/test_home_page.py`, `test_admin_page.py`, `test_classify_page.py` | Pages work with Streamlit's `AppTest`, using fake services |
| Manual | [quickstart.md](quickstart.md) | Real models and the assignment's demonstration |

Integration tests carry the `integration` marker. They skip with a clear reason if MinIO is unreachable locally, and CI
treats MinIO as required, so skipping can't hide a failure there.

## Docker and Local Development

- **Default**: `docker compose up -d minio minio-console` starts MinIO and its console, and Streamlit runs in the local virtual environment.
- **Optional container**: `docker compose --profile app up` also runs the application from the `Dockerfile`.
- **Ports**: 9000 for the MinIO API, 9001 for its console, and 8501 for Streamlit.
- **Commands**: The `Makefile` wraps common commands, and the README also lists them directly because `make` often
  isn't installed on Windows.

**Validated in T003**: The MinIO and console images start and work together, and the underscore bucket name is
rejected, as recorded in research R2 and R3.

## Deployment

The target is local development and trusted environments only, because the admin page has no authentication. CI
validates each change, but the project has no production deployment. The default MinIO credentials and image are
local-only.

## Project Structure

### Documentation (this feature)

```text
specs/001-zero-shot-platform/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── prediction-record.schema.json
│   ├── config-files.md
│   └── ui-pages.md
├── checklists/
│   └── requirements.md
└── tasks.md             # Created by /speckit-tasks
```

### Source Code

```text
zeroshot-textclassification-mlops/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── configs/
│   ├── models.yaml
│   └── labels.yaml
├── app/
│   ├── streamlit_app.py
│   ├── ui_state.py
│   ├── pages/
│   │   ├── 1_Admin.py
│   │   └── 2_Classify_Text.py
│   ├── services/
│   │   ├── model_manager.py
│   │   ├── label_manager.py
│   │   ├── classifier.py
│   │   └── minio_storage.py
│   └── utils/
│       ├── config.py
│       ├── errors.py
│       └── logging_config.py
├── models/
│   └── .gitkeep
└── tests/
    ├── unit/
    └── integration/

.github/workflows/zeroshot-ci.yml   # At the repository root (R9)
```

**Structure Decision**: This is the assignment's recommended single-project layout, inside the existing project folder
rather than a new `zeroshot-mlops/`. It adds `label_manager.py`, `errors.py`, and `ui_state.py` to keep responsibilities
separate, splits tests by level, and moves the CI workflow to the repository root because GitHub requires it.

## Complexity Tracking

No constitution violations need justifying. Departures from the assignment's examples are recorded in constitution
decisions DR-001 to DR-003, and in [research.md](research.md).
