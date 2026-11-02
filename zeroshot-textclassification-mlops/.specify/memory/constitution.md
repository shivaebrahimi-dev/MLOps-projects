# Zero-Shot Text Classification Platform Constitution

## Core Principles

### I. Project Purpose and Scope

The project MUST deliver a reproducible, Python-based zero-shot text-classification platform
using PyTorch, Hugging Face Transformers, Streamlit, and MinIO, for learning and demonstrating
MLOps practices.

The first version MUST include:

- An administrator workflow for managing models and candidate labels.
- A separate user workflow for classifying text.
- Support for at least two zero-shot models, including
  `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`.
- Persistence of every successful classification to MinIO.
- The supporting MLOps practices required by this constitution: environment-based configuration,
  structured logging, automated tests, Docker Compose support for MinIO, CI validation, and
  documentation.

The following features are explicitly out of scope for the first version: admin authentication,
Prometheus metrics, MinIO lifecycle policies, a prediction-history page, CSV export, batch
classification, and model comparison. They MAY be added only after every acceptance criterion in
the specification passes. A Docker image for the Streamlit application, a status panel, and
inference-latency measurement are optional, and the implementation plan decides whether to include
them.

### II. Separation of Concerns

Streamlit UI, model management, inference, configuration, logging, and MinIO storage MUST remain
separate components.

- Streamlit pages MUST contain only interface logic and MUST delegate all other work to services.
- Services MUST NOT import Streamlit, so that they can be tested without the UI.

### III. Reproducibility

Another developer MUST be able to reproduce the application using only the repository and its
setup instructions.

- The Python version, Python dependencies, PyTorch version, Transformers version, model
  identifiers, model revisions, and MinIO configuration, including the container image version,
  MUST be pinned or clearly documented.
- Each stored prediction MUST record the model identifier and, when available, the revision used.

### IV. Configuration and Secrets

- Credentials and environment-specific settings MUST come from environment variables or secret
  management. Non-secret settings MAY come from documented YAML configuration files.
- Adding a supported model MUST require a configuration change, not a major code change.
- Secrets MUST NOT be hard-coded or committed. `.env.example` MUST contain placeholder values only.

### V. Model Lifecycle Management

- Models MUST be downloaded from Hugging Face and stored in a controlled, configurable local
  directory.
- A model that is already available locally MUST be reused, not downloaded again.
- A partial or failed download MUST NOT be treated as ready and MUST NOT leave an invalid model
  directory behind.
- Deleting a model and clearing the Hugging Face cache MUST be explicit administrator actions.
- Multiple large models MUST NOT be loaded into memory unnecessarily, and initialized pipelines
  SHOULD be cached where appropriate.

### VI. Data Persistence and Storage Contract

- Every successful classification MUST be saved as structured JSON in the configured MinIO bucket.
- MinIO is the mandatory object store and the system of record for classification outputs.
  Results MUST NOT be stored in another object-storage service or only on the local filesystem.
- Local files MAY be used for temporary processing, caching, or testing, but MUST NOT replace MinIO.
- The bucket name MUST be configurable through an environment variable, and its default MUST be
  `zeroshot-learning-output` (see DR-001). The application MUST create the bucket automatically
  when it does not exist.
- Each stored prediction MUST have a unique request ID.

### VII. Testing Standards

- Model management, inference, and storage services MUST have unit tests.
- MinIO behavior MUST have integration tests, and at least one end-to-end test MUST verify that a
  prediction can be written to MinIO.
- Tests that would require large model downloads MUST use mocks or lightweight fixtures whenever
  possible.

### VIII. Error Handling

Invalid input, unavailable models, failed downloads, and MinIO failures MUST produce clear,
user-facing errors and MUST NOT expose secrets. The application MUST validate and sanitize user
input and MUST degrade gracefully when a model or MinIO is unavailable.

### IX. Observability

Model downloads, inference, cache cleanup, and MinIO writes MUST be logged as structured entries
with useful metadata, such as request ID, model identifier, duration, and outcome.

### X. Security and Privacy

- Logs MUST NOT contain credentials or tokens, and MUST NOT contain classification input text
  unless it is needed to diagnose a failure.
- Model files, cache directories, `.env` files, and credentials MUST NOT be committed to Git.
- If admin authentication is not implemented, the README MUST state that the admin page is
  intended only for local development or trusted environments.

### XI. Documentation

The README MUST be written for a developer new to the project and MUST explain setup, usage,
architecture, model caching and deletion, candidate labels, MinIO storage, testing, troubleshooting,
and security limitations. It MUST be reviewed against the actual implementation and updated whenever
the code structure or setup process changes.

## Technology Standards and Scope

- **Programming language**: The project MUST be implemented in Python. Other languages MAY be used
  only for supporting configuration, infrastructure, or deployment files, such as YAML, JSON,
  Dockerfiles, or shell scripts.
- **Deep-learning framework**: PyTorch MUST be used for model loading and inference. No other
  deep-learning framework may be introduced without an approved decision record.
- **Framework interoperability**: Models MAY be loaded through Transformers, but inference MUST run
  on the PyTorch backend. The project MUST NOT silently switch to another backend.
- **Local infrastructure**: MinIO MUST start locally through Docker Compose.
- **Technology consistency**: Any replacement of or addition to the core stack (Python, PyTorch,
  Hugging Face Transformers, Streamlit, MinIO) MUST have a decision record explaining the reason,
  benefits, risks, and impact on reproducibility. Supporting libraries MUST be pinned and listed in
  DR-002.

## Quality Gates and Compliance

The project MUST pass all of these checks before it is considered complete:

- Formatting or lint checks pass.
- The automated test suite passes in CI on every pull request and before final submission. MinIO
  integration tests pass when MinIO is running.
- All acceptance criteria in the specification are satisfied.
- Configuration loads only from environment variables or documented configuration files.
- No model artifacts, cache directories, `.env` files, or credentials are tracked by Git.
- At least one end-to-end test verifies that a prediction is written to MinIO.
- The README explains how a new developer can reproduce the project locally.

## Governance

This constitution is the highest-level project agreement and takes precedence over other project
documents. The specification, plan, data model, and tasks MUST be checked for consistency with it
before implementation begins.

- **Spec Kit workflow**: Artifacts MUST be created with Spec Kit before implementation, in this
  order: constitution, specification, implementation plan, data model, task breakdown, then task
  implementation. The constitution MUST be reviewed before the other artifacts are created. The
  plan and data model MAY be generated in the same Spec Kit step. All artifacts MUST be included in
  the repository.
- **Amendments**: A change MUST update this file and state its reason and impact.
- **Exceptions**: A deliberate exception to a principle MUST be added to Decision Records with its
  reason and impact, and approved by the project developer, before the exception is implemented.
- **Versioning**: MAJOR for removing or redefining a principle, MINOR for adding a principle or
  materially expanding guidance, and PATCH for clarifications and wording fixes.
- **Compliance review**: Each pull request and the final submission MUST be checked against the
  Quality Gates.

## Decision Records

### DR-001: Default bucket name uses hyphens instead of underscores

- **Status**: Accepted by the project developer, 2026-11-01.
- **Decision**: The default bucket name is `zeroshot-learning-output`, not the
  `zeroshot_learning_output` named in the original assignment.
- **Reason**: A running MinIO server answers `InvalidBucketName` for the underscore name, and the Python
  client and `mc` tool also refuse it. All of them accept the hyphen name.
- **Impact**: This differs from the original assignment's bucket name, which the assignment's notice
  permits because the main project objective is preserved. The name remains configurable through
  `MINIO_BUCKET`, and the difference MUST be disclosed in the README and the final submission.

### DR-002: Approved supporting libraries

- **Status**: Accepted by the project developer, 2026-11-01.
- **Decision**: The core stack MAY be supported by the `minio` client, `urllib3`, `huggingface_hub`,
  `PyYAML`, `python-dotenv`, `pytest`, a lint or formatting tool, and tokenizer dependencies that a
  supported model requires. A library not listed here MUST be added to this record before it is used.
- **Reason**: The core stack needs these libraries for object storage, short storage timeouts, model
  downloads, YAML configuration, loading `.env` files, testing, quality gates, and tokenization. None of
  them replaces a core technology.
- **Benefits and risks**: They avoid custom implementations of common tasks, but each one adds a
  dependency that can change or conflict with others.
- **Impact on reproducibility**: Each library MUST be pinned with the project's other dependencies.

### DR-003: MinIO container image source

- **Status**: Accepted by the project developer, 2026-11-01. The console image was added during task
  T003 because this server image serves no console, and the assignment's sections 7 and 13 require one.
- **Decision**: Local MinIO uses the pinned image `bitnamilegacy/minio:2025.7.23-debian-12-r5`, and its
  console uses `bitnamilegacy/minio-object-browser:2.0.2-debian-12-r4`, instead of the `minio/minio`
  image in the assignment's example. They are configurable through `MINIO_IMAGE` and
  `MINIO_CONSOLE_IMAGE`.
- **Reason**: On 2026-11-01, the official images on Docker Hub and Quay were unavailable, and MinIO's
  repository states that the community edition is now distributed as source only. The chosen images were
  pulled and tested: storage, restarts, console login, and bucket access all worked.
- **Benefits and risks**: The pinned image is a packaged MinIO server that should stay unchanged, but
  it receives no further updates and its layout differs from the official image. It MUST be used only
  for local development and testing.
- **Impact on reproducibility**: A fixed tag improves repeatability. If the image becomes unavailable,
  `MINIO_IMAGE` allows a replacement, which MUST be recorded here.

**Version**: 1.0.3 | **Ratified**: 2026-11-01 | **Last Amended**: 2026-11-01
