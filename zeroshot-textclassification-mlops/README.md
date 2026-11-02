# Zero-Shot Text Classification Platform

A small MLOps project that classifies text into categories you choose at request time, without training a model. An administrator downloads and manages Hugging Face models and label sets, a user submits text and reads ranked results, and every successful result is saved as JSON in MinIO.

The project was built spec-first with [Spec Kit](https://github.com/github/spec-kit). Its objectives are to practice AI-assisted, spec-driven development and basic MLOps: reproducibility, model lifecycle management, testing, logging, and clear separation of responsibilities.

**Contents**

- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Code structure](#code-structure)
- [Models](#models)
- [Model download, caching, deletion, and cleanup](#model-download-caching-deletion-and-cleanup)
- [Candidate labels](#candidate-labels)
- [MinIO storage](#minio-storage)
- [Configuration](#configuration)
- [Testing](#testing)
- [Stopping and restarting](#stopping-and-restarting)
- [Running the app in a container](#running-the-app-in-a-container)
- [Troubleshooting](#troubleshooting)
- [Security limitations](#security-limitations)
- [Differences from the assignment](#differences-from-the-assignment)
- [Spec Kit documents](#spec-kit-documents)

## Quick start

### Prerequisites

- Python 3.12
- Docker Desktop, running
- Internet access to download models
- Disk space: about 400 MB for DeBERTa and 1.7 GB for BART, plus several GB for the Python packages and Docker images

Classification runs on CPU, and a GPU is optional. The commands below are for PowerShell on Windows. On macOS or Linux, activate the environment with `source .venv/bin/activate` and copy the settings file with `cp .env.example .env`.

### 1. Install

```powershell
git clone https://github.com/shivaebrahimi-dev/MLOps-projects.git
cd MLOps-projects/zeroshot-textclassification-mlops
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

### 2. Start MinIO

```powershell
docker compose up -d --wait minio minio-console
docker compose ps
```

Expected: both services show as `Up`, and `minio` shows as `healthy`.

### 3. Start the application

```powershell
streamlit run app/streamlit_app.py
```

Expected: the app opens at <http://localhost:8501>.

### 4. Download a model

1. Open **Admin** in the sidebar.
2. Under **Download**, select **DeBERTa Zero-Shot v2**, then click **Download selected**.

Expected: a success message appears, and the model's status changes to **Ready**. This took about 40 seconds during development, though it depends on your connection.

### 5. Classify text

1. Open **Classify Text** in the sidebar.
2. Enter `The team won the championship after a very close game.`
3. Keep the default model and label set, then click **Classify**.

Expected: your text and the predicted class `sports`, with its confidence, a ranked list of labels, the model details, and a message showing where the result was saved. On CPU during development, the first DeBERTa request took about 8 seconds because the model must load, and later ones took 2 to 3 seconds. BART took about 0.6 seconds per request after loading.

### 6. Verify the result in MinIO

1. Open the console at <http://localhost:9001>.
2. Log in with `minioadmin` / `minioadmin`, or your own values from `.env`.
3. Open the `zeroshot-learning-output` bucket, then browse to the path shown on the classification page, under `predictions/YYYY/MM/DD/`.

Expected: a JSON object named after the request ID, in the [format below](#minio-storage).

## Architecture

```text
Streamlit pages (interface only)
  ├── Admin page ───────────┐
  └── Classify Text page ───┤
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

**How the parts work together**

- The Streamlit pages only handle the interface. They call services and display the results or errors those services return, and a test fails if a service imports Streamlit.
- The admin page uses `ModelManager` to download and delete models and `LabelManager` to edit label sets.
- The classification page lists models from `ModelManager` and label sets from `LabelManager`, sends the request to `Classifier`, and passes the result to `PredictionStorage`.
- `Classifier` asks `ModelManager` whether a model is ready and loads it only from the local model directory.
- `PredictionStorage` is the only code that talks to MinIO.

## Code structure

```text
zeroshot-textclassification-mlops/
├── app/
│   ├── streamlit_app.py          Home page
│   ├── ui_state.py               Creates the shared services
│   ├── pages/
│   │   ├── 1_Admin.py            Model, cache, and label management
│   │   └── 2_Classify_Text.py    Text input and results
│   ├── services/
│   │   ├── model_manager.py      Model configuration, status, download, deletion, cache cleanup
│   │   ├── label_manager.py      Label validation and persistence
│   │   ├── classifier.py         Input validation, pipeline loading, inference
│   │   └── minio_storage.py      Bucket creation, record building, writes to MinIO
│   └── utils/
│       ├── config.py             Loads and validates settings
│       ├── errors.py             Typed errors with messages that are safe to display
│       └── logging_config.py     Structured JSON logs without secrets or input text
├── configs/
│   ├── models.yaml               Supported models and pinned revisions
│   └── labels.yaml               Saved label sets
├── models/                       Downloaded models (ignored by Git)
├── tests/
│   ├── unit/                     Fast tests using fakes
│   └── integration/              Tests against a running MinIO and Hugging Face
├── specs/001-zero-shot-platform/ Specification, plan, data model, and tasks
├── .specify/memory/constitution.md  Project principles and decision records
├── docker-compose.yml            MinIO, its console, and the optional app container
├── Dockerfile                    Optional app container
├── requirements.txt              Pinned dependencies
├── pyproject.toml                Project, ruff, and pytest settings
└── .env.example                  Template for your local settings
```

The CI workflow is in `.github/workflows/zeroshot-ci.yml` at the repository root, because GitHub only runs workflows from there.

## Models

| Name | Hugging Face identifier | Pinned revision | Weights |
|---|---|---|---|
| DeBERTa Zero-Shot v2 | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | `8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8` | 369 MB |
| BART MNLI | `facebook/bart-large-mnli` | `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce` | 1.63 GB |

**To add a model**, add an entry to `configs/models.yaml` and restart the app. No code change is needed, but the repository must provide `model.safetensors`, `config.json`, and a tokenizer.

```yaml
models:
  - id: org/model-name
    name: Display name
    task: zero-shot-classification
    revision: full-commit-hash
```

## Model download, caching, deletion, and cleanup

**Download**

1. If the model is already ready, nothing is downloaded and the page says so.
2. A lock file prevents a second download of the same model from starting.
3. The pinned revision downloads into a temporary folder under `models/.tmp/`, including only configuration, tokenizer, and `model.safetensors` files.
4. The required files are checked, and a completion marker, `.download-complete.json`, is written with the model ID, revision, and time.
5. The folder is renamed to `models/{org}/{name}`.

Because of this order, an interrupted or failed download can never look ready. Temporary files are removed, the failure is shown on the admin page, and you can retry.

**Status**

| Status | Meaning |
|---|---|
| Not downloaded | No local copy exists. |
| Downloading | A download is running. |
| Ready | The folder, its completion marker, and its required files all exist. |
| Failed | The last download failed, or a local copy is incomplete. Retry the download, or delete the entry. |

**Deletion**: Choose the model on the admin page and tick the confirmation box. The model is released from memory, then its files are removed.

**Cache cleanup**: Tick the confirmation box and click **Clear cache**. This removes only the `hub` and `xet` folders under `HF_HOME`. It keeps saved Hugging Face login tokens and the models in `models/`.

> Models download straight to `models/` rather than through the shared Hugging Face cache, which avoids storing each large model twice. Clearing the cache may therefore report little or nothing to remove, which is expected.

Only one model stays in memory at a time. Selecting a different model replaces the one that is loaded.

## Candidate labels

Label sets are stored in `configs/labels.yaml`. On the admin page, under **Candidate labels**, you can choose a set or **New label set**, add and remove labels, and click **Save label set**. Saved sets appear on the classification page.

A label set needs at least two labels. Empty labels and duplicates are rejected, and labels that differ only by case or surrounding spaces count as duplicates. Control characters are removed from labels and input text.

> Saving from the admin page changes `configs/labels.yaml`, which Git tracks. Check that change before committing it. To undo it, run `git restore configs/labels.yaml`.

## MinIO storage

The default bucket is `zeroshot-learning-output`, which you can change with `MINIO_BUCKET`. The app creates it automatically when the first result is saved.

Each successful classification is stored as one JSON object at:

```text
predictions/YYYY/MM/DD/{request_id}.json
```

The date is the UTC date of the prediction. An example record:

```json
{
  "request_id": "7c3cf173-aaff-4265-83b0-d08ce501847f",
  "timestamp": "2026-11-02T01:16:36Z",
  "input_text": "The team won the championship after a very close game.",
  "predicted_class": "sports",
  "confidence": 0.9995814561843872,
  "candidate_labels": ["sports", "politics", "technology", "finance"],
  "ranked_predictions": [
    {"label": "sports", "score": 0.9995814561843872},
    {"label": "technology", "score": 0.00019482917559798807},
    {"label": "politics", "score": 0.00011408763384679332},
    {"label": "finance", "score": 0.00010971700976369902}
  ],
  "model_name": "DeBERTa Zero-Shot v2",
  "model": "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
  "model_revision": "8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8",
  "application_version": "0.1.0",
  "session_id": "b6f1f0f5-5a53-4d9e-9a51-3f0c7f7a2c11",
  "latency_ms": 5149.7
}
```

The first request's `latency_ms` includes loading the model. The formal format is in [prediction-record.schema.json](specs/001-zero-shot-platform/contracts/prediction-record.schema.json).

If MinIO is unavailable, you still see the result, along with a clear error that it was not saved and its request ID. Nothing is written to local files instead, because MinIO is the system of record.

## Configuration

Settings come from environment variables, or from your local `.env` file, which Git ignores.

| Variable | Default | Purpose |
|---|---|---|
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO address |
| `MINIO_ACCESS_KEY` | Required | MinIO access key |
| `MINIO_SECRET_KEY` | Required | MinIO secret key |
| `MINIO_SECURE` | `false` | Use HTTPS when `true` |
| `MINIO_BUCKET` | `zeroshot-learning-output` | Bucket for prediction records |
| `MINIO_IMAGE` | `bitnamilegacy/minio:2025.7.23-debian-12-r5` | MinIO image for Docker Compose |
| `MINIO_CONSOLE_IMAGE` | `bitnamilegacy/minio-object-browser:2.0.2-debian-12-r4` | Console image for Docker Compose |
| `MODEL_DIR` | `./models` | Where models are stored |
| `HF_HOME` | `./.cache/huggingface` | Hugging Face cache location |
| `HF_TOKEN` | Empty | Optional token for restricted models |
| `MODELS_CONFIG` | `configs/models.yaml` | Model configuration file |
| `LABELS_CONFIG` | `configs/labels.yaml` | Label configuration file |
| `MAX_INPUT_CHARS` | `10000` | Maximum length of input text |
| `APP_VERSION` | `0.1.0` | Version stored with each prediction |
| `LOG_LEVEL` | `INFO` | Logging level |

Relative paths are relative to the project folder.

## Testing

Run these with the virtual environment active:

```powershell
ruff check .
ruff format --check .
pytest -m "not integration"
docker compose up -d --wait minio
pytest -m integration
```

Expected: every test passes.

- **Unit tests** use fakes and need neither MinIO nor large downloads. They cover model management, labels, classification, storage, configuration, logging, and the pages.
- **Integration tests** need a running MinIO and internet access for a tiny test model. They create uniquely named buckets and remove them afterward.
- If MinIO is unreachable, integration tests are skipped with a reason. Set `REQUIRE_MINIO=true` to make them fail instead, as CI does.

The `Makefile` wraps these commands if you have `make` installed. CI runs the same checks on Linux with CPU-only PyTorch whenever this project changes.

## Stopping and restarting

```powershell
docker compose stop      # Stop services and keep their data
docker compose start     # Start them again
docker compose down      # Remove containers and keep stored predictions
docker compose down -v   # Remove containers and delete stored predictions
```

To stop Streamlit, press `Ctrl+C` in its terminal.

## Running the app in a container

This is optional. It builds an image of about 2.3 GB that uses CPU-only PyTorch.

```powershell
docker compose --profile app up -d --build
```

The app opens at <http://localhost:8501>. Its models are kept in a Docker volume rather than your local `models/` folder, so models you downloaded locally aren't visible to it. To stop it, run `docker compose --profile app down`.

## Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| `Filename too long` during `git clone` | The clone folder's path is too long for Windows. Clone to a short path such as `C:\src`. |
| `Activate.ps1` is blocked by an execution policy | Run `Set-ExecutionPolicy -Scope Process RemoteSigned`, then activate again. |
| `streamlit` or `pytest` is not recognized | The virtual environment isn't active. Activate it. |
| `MINIO_ACCESS_KEY is not set` | There's no `.env` file. Run `Copy-Item .env.example .env`, then restart Streamlit. |
| `failed to connect to the docker API` | Docker Desktop isn't running. Start it and wait until it's ready. |
| `ports are not available` or `address already in use` | Another program is using port 9000, 9001, or 8501. Stop it, or use a different Streamlit port with `--server.port`. |
| "MinIO could not be reached" | MinIO isn't running. Run `docker compose up -d --wait minio minio-console`. |
| "MinIO refused the credentials" | `MINIO_ACCESS_KEY` or `MINIO_SECRET_KEY` doesn't match what MinIO started with. MinIO keeps its first credentials in its volume, so restore the original values, or run `docker compose down -v`, which also deletes stored predictions. |
| "not a valid bucket name" | `MINIO_BUCKET` contains characters such as underscores. Use lowercase letters, numbers, and hyphens. |
| Console opens but shows no bucket | The bucket is created when the first result is saved. Classify some text first. |
| "No model is ready yet" | Download a model on the admin page. |
| A model shows **Failed** | Read the **Details** column. Retry the download, or delete the entry. |
| A download fails | Check your connection and free disk space. Restricted models also need `HF_TOKEN`. |
| "could not classify this text" | The computer may be short of memory, especially with BART. Close other programs. If it continues, delete and download the model again. |
| A model can't be deleted | Something is using its files. Close it and try again. |
| "model did not load on the required PyTorch backend" | The model loaded on something other than PyTorch. Reinstall using `requirements.txt`. |

Application logs are JSON lines in the terminal that runs Streamlit. They use event names such as `inference_completed`, `prediction_stored`, and `model_download_failed`.

## Security limitations

- **No login**: The admin page has no authentication, so anyone who can reach the app can download and delete models. It's intended only for local development or trusted environments.
- **Local access**: Streamlit and the Docker ports are limited to this computer by default. Don't expose them to a network without adding authentication.
- **Default credentials**: `minioadmin` is for local development only. Change it before using real data.
- **Stored text**: MinIO stores the full input text of every prediction, so don't submit sensitive text. Logs leave input text out.
- **Secrets**: `.env` files, tokens, downloaded models, and caches are ignored by Git and must never be committed.

## Differences from the assignment

The assignment describes itself as a draft that developers may adapt as long as its main objective is preserved. This project differs from its examples in these ways:

| Assignment example | This project | Reason |
|---|---|---|
| Bucket `zeroshot_learning_output` | `zeroshot-learning-output` | MinIO's server rejects the original name with `InvalidBucketName`. |
| Image `minio/minio` | Pinned Bitnami server and console images | The official images were unavailable during development, and this server image has no built-in console. |
| `TRANSFORMERS_CACHE` | `HF_HOME` only | The Transformers cache code now uses `HF_HOME`. |
| `MINIO_ENDPOINT=minio:9000` in `.env.example` | `localhost:9000` | Streamlit runs on your computer by default, where `minio` doesn't resolve. The optional app container uses `minio:9000`. |
| Project folder `zeroshot-mlops/` | `zeroshot-textclassification-mlops/` | The repository also holds another assignment, so the project has a descriptive folder. |
| `ci.yml` | `zeroshot-ci.yml` | The name identifies this project in a shared repository. |
| Flat `tests/` folder | `tests/unit/` and `tests/integration/` | This lets the fast tests run without MinIO. |
| CI workflow in the project folder | Repository root | GitHub only runs workflows from there. |
| `specification.md` | `spec.md` | This is the file name Spec Kit generates. |
| GitHub Copilot for AI assistance and this README | Claude Code | This was the AI coding assistant used with Spec Kit. The README was reviewed against the implementation. |

The first two are recorded with their evidence as decisions DR-001 and DR-003 in the [constitution](.specify/memory/constitution.md).

## Spec Kit documents

The documents were created before the code, in this order:

1. [Constitution](.specify/memory/constitution.md)
2. [Specification](specs/001-zero-shot-platform/spec.md)
3. [Implementation plan](specs/001-zero-shot-platform/plan.md), with [research notes](specs/001-zero-shot-platform/research.md)
4. [Data model](specs/001-zero-shot-platform/data-model.md), with [contracts](specs/001-zero-shot-platform/contracts/)
5. [Task breakdown](specs/001-zero-shot-platform/tasks.md)
6. Implementation, with a [validation guide](specs/001-zero-shot-platform/quickstart.md)
