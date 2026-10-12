# Fashion-MNIST MLOps: train, track, store and serve

An end-to-end MLOps application that trains a LeNet-5-style model on Fashion-MNIST, tracks the experiment in MLflow, stores the trained model in MinIO, sends a training-complete webhook, and serves single-image predictions through a FastAPI backend with a Streamlit UI. Every service runs with Docker Compose.

```text
configs/train.toml ──► trainer ──► MLflow   (params, metrics, model reference)
                          │
                          ├──────► MinIO    fashion-minst-bucket/model/<run-id>/model.pt
                          │
                          └──────► webhook  (only after the upload succeeded)

browser ──► Streamlit UI ──► FastAPI /predict ──► loads the model from MinIO
```

## Contents

- [Quick start](#quick-start)
- [Services and ports](#services-and-ports)
- [Configuration](#configuration)
- [Training](#training)
- [MLflow experiment tracking](#mlflow-experiment-tracking)
- [MinIO model storage](#minio-model-storage)
- [Training-complete webhook](#training-complete-webhook)
- [Prediction API](#prediction-api)
- [Streamlit UI](#streamlit-ui)
- [Running without Docker](#running-without-docker)
- [Tests](#tests)
- [Project layout](#project-layout)

## Quick start

Requirements: Docker with Docker Compose. Nothing else needs to be installed.

```bash
# 1. Create your local environment file and set a MinIO password in it
cp .env.example .env            # PowerShell: Copy-Item .env.example .env

# 2. Start storage, tracking and the local webhook receiver
docker compose up -d minio mlflow webhook-receiver

# 3. Train. This logs to MLflow, uploads the model to MinIO and sends the webhook
docker compose run --rm trainer

# 4. Start the prediction API and the UI
docker compose up -d api ui
```

Then open <http://localhost:8501>, upload one of the images in [samples/](samples/) and press **Predict**.

Step 3 must finish before predictions work, because the API serves the model that training uploads. If you start the API and UI first, nothing breaks: the UI shows "No trained model is available yet" and starts working as soon as a training run completes, with no restart needed.

To stop everything: `docker compose down` (add `-v` to also delete the stored models and MLflow history).

## Services and ports

| Service | URL | Purpose |
|---|---|---|
| `ui` | <http://localhost:8501> | Streamlit prediction interface |
| `api` | <http://localhost:8000> (docs at `/docs`) | FastAPI prediction backend |
| `mlflow` | <http://localhost:5000> | Experiment tracking UI |
| `minio` | <http://localhost:9001> (console), `:9000` (S3 API) | Model storage |
| `webhook-receiver` | <http://localhost:8090> | Prints received webhooks, for testing |
| `trainer` | none | One-shot training job (`docker compose run --rm trainer`) |

If a port is already taken on your machine, change `API_PORT`, `UI_PORT` or `WEBHOOK_PORT` in `.env`.

MinIO and MLflow keep their data in the named volumes `minio-data` and `mlflow-data`, so models and runs survive restarts.

## Configuration

### Secrets and endpoints: `.env`

Credentials and service addresses come only from environment variables. `.env.example` lists all of them with placeholder values; the real `.env` is gitignored and is also excluded from the Docker images.

| Variable | Default | Meaning |
|---|---|---|
| `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` | none (required) | MinIO credentials, also used as the server's root user |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO address (`minio:9000` inside Compose) |
| `MINIO_BUCKET` | `fashion-minst-bucket` | Model bucket |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server (`http://mlflow:5000` inside Compose) |
| `WEBHOOK_URL` | `http://localhost:8090/webhook` | Webhook target when training on the host |
| `TRAINER_WEBHOOK_URL` | empty | Webhook target for the Compose trainer; empty means the bundled receiver |
| `MODEL_OBJECT_KEY` | empty | Pin the API to one model; empty means "latest" |
| `API_URL` | `http://localhost:8000` | Where the UI finds the API (`http://api:8000` inside Compose) |

### Training settings: `configs/train.toml`

No training setting is hard-coded. Each one is resolved in this order, highest priority first:

1. command-line flag, for example `--epochs 10`
2. environment variable with a `TRAIN_` prefix, for example `TRAIN_EPOCHS=10`
3. the TOML file passed with `--config`
4. the built-in default

| Setting | Default | Notes |
|---|---|---|
| `epochs` | `5` | |
| `batch_size` | `64` | |
| `learning_rate` | `0.001` | |
| `optimizer` | `adam` | `adam` or `sgd` |
| `loss` | `cross_entropy` | `cross_entropy` or `nll` (log-softmax followed by negative log-likelihood) |
| `seed` | `42` | Seeds the split, the weight initialisation and batch shuffling |
| `val_split` | `0.1` | Fraction of the 60,000 training images held out for validation |
| `momentum` | `0.9` | Optional, used by `sgd` only |
| `weight_decay` | `0.0` | Optional |
| `early_stopping_patience` | `0` | Optional. Stop after this many epochs without validation-loss improvement and restore the best weights. `0` disables it |
| `num_workers` | `0` | Optional, DataLoader worker processes |
| `data_dir` | `data` | Where Fashion-MNIST is downloaded |
| `experiment_name` | `fashion-mnist-lenet5` | MLflow experiment |

Invalid values (for example `optimizer = "rmsprop"`) and unknown keys are rejected before training starts.

## Training

```bash
# Use the example configuration
docker compose run --rm trainer

# Override settings on the command line
docker compose run --rm trainer --config configs/train.toml --epochs 10 --optimizer sgd --learning-rate 0.01

# Or through environment variables
docker compose run --rm -e TRAIN_EPOCHS=10 -e TRAIN_SEED=7 trainer
```

`configs/` is mounted into the container, so edits to `configs/train.toml` apply without rebuilding the image.

The model is LeNet-5 adapted to 28 × 28 grayscale input: two 5 × 5 convolutions (6 and 16 channels, each followed by ReLU and 2 × 2 max pooling) and three fully connected layers (120, 84, 10). With the default configuration it reaches about 90% validation accuracy in five epochs, which takes well under a minute on a CPU.

A run does these things in order, and stops at the first failure:

1. checks that MLflow and MinIO are reachable and creates the bucket if needed
2. trains, logging metrics to MLflow after each epoch
3. uploads the model to MinIO
4. records the MinIO reference in MLflow
5. sends the webhook

If a run fails, the trainer exits with a non-zero status, prints the reason, and the MLflow run is marked `FAILED`.

## MLflow experiment tracking

Open <http://localhost:5000>, choose the **fashion-mnist-lenet5** experiment and click a run.

| Logged as | What |
|---|---|
| Parameters | Every training setting from the table above, as actually used by the run |
| Metrics per epoch | `train_loss`, `val_loss`, `val_accuracy` (see the **Model metrics** tab for the curves) |
| Final metrics | `final_val_accuracy`, `final_val_loss`, `epochs_run` |
| Tags | `model_bucket`, `model_object_key`, `webhook_status` (`sent`, `skipped` or `failed`) |
| Artifact `config.json` | The resolved configuration in the same shape as `configs/train.toml` |
| Artifact `model_info.json` | Bucket, object key, `s3://` URI, size, SHA-256, class names and validation accuracy |

To reproduce a run, download its `config.json`, convert it to TOML or pass its values as flags, and train again: the seed fixes the split, the initial weights and the batch order.

MLflow stores a reference to the model rather than a second copy; the model itself lives in MinIO.

## MinIO model storage

The bucket is named **`fashion-minst-bucket`**. The spelling `minst` is intentional: it is the exact name the assignment specifies.

```text
fashion-minst-bucket/
└── model/
    ├── <mlflow-run-id>/model.pt     one per training run, never overwritten
    ├── <mlflow-run-id>/model.pt
    └── latest.json                  points at the most recently uploaded model
```

- The object key contains the MLflow run ID, so later runs never overwrite earlier models and every model can be traced back to its run.
- The bucket is created automatically by the trainer if it does not exist.
- `model.pt` holds the weights and class names. It is loaded with `weights_only=True`, so a model file cannot execute code when loaded.

To browse the bucket, open <http://localhost:9001> and log in with `MINIO_ACCESS_KEY` and `MINIO_SECRET_KEY` from your `.env`.

### Which model does the API serve?

- **Default:** the newest one. The API reads `model/latest.json`, which the trainer rewrites after each successful upload. The API re-reads it on each request, so a new training run is picked up without a restart.
- **Pinned:** set `MODEL_OBJECT_KEY` in `.env`, for example `MODEL_OBJECT_KEY=model/<run-id>/model.pt`, then run `docker compose up -d api`. The key for any run is shown in MLflow (tag `model_object_key`) and in the webhook payload.

If the object or bucket is missing, or MinIO is unreachable, the API answers `503` with a message saying which of these happened. A model that is already loaded keeps serving during a MinIO outage.

The Compose file uses the Chainguard build of MinIO (`cgr.dev/chainguard/minio`) because MinIO no longer publishes the `minio/minio` image on Docker Hub.

## Training-complete webhook

After a run has trained, logged to MLflow **and** uploaded its model, the trainer POSTs a JSON notification to `WEBHOOK_URL`. If training, MLflow logging or the upload fails, no notification is sent.

Sample payload ([samples/sample_webhook_payload.json](samples/sample_webhook_payload.json)):

```json
{
  "event": "training_complete",
  "status": "success",
  "run_id": "ed7d742923e94018ab792193064917ba",
  "experiment": "fashion-mnist-lenet5",
  "model": {
    "bucket": "fashion-minst-bucket",
    "object_key": "model/ed7d742923e94018ab792193064917ba/model.pt",
    "uri": "s3://fashion-minst-bucket/model/ed7d742923e94018ab792193064917ba/model.pt"
  },
  "metrics": {
    "val_accuracy": 0.8735,
    "val_loss": 0.33607731509208677,
    "epochs_run": 2
  },
  "timestamp": "2026-10-12T17:58:43+00:00"
}
```

### Testing notifications

The Compose stack includes `webhook-receiver`, a small server that prints whatever it receives. After a training run:

```bash
docker compose logs webhook-receiver
```

To send the notification somewhere else (for example a <https://webhook.site> URL), set `TRAINER_WEBHOOK_URL` in `.env`.

To see the failure behaviour, stop MinIO while a run is training (`docker compose stop minio`): the trainer exits with an error, the MLflow run is `FAILED`, and no webhook arrives.

If the model was stored but the receiver could not be reached, the trainer says so explicitly, tags the run `webhook_status=failed` and exits with status 2. The stored model remains usable.

## Prediction API

Interactive documentation is at <http://localhost:8000/docs>.

### `GET /health`

Returns `200` whenever the service is running. `model_available` says whether predictions are possible.

```json
{"status": "ok", "model_available": true, "model_object_key": "model/<run-id>/model.pt", "detail": null}
```

### `POST /predict`

Request: `multipart/form-data` with one field named `file` containing a single image (PNG, JPEG, BMP, WebP, ...; at most 5 MB).

```bash
curl -F "file=@samples/9_ankle-boot.png" http://localhost:8000/predict
```

Response:

```json
{
  "predicted_class": "Ankle boot",
  "class_index": 9,
  "confidence": 0.958,
  "probabilities": {"T-shirt/top": 0.0001, "Trouser": 0.0, "...": 0.0, "Ankle boot": 0.958},
  "model_object_key": "model/<run-id>/model.pt"
}
```

The image is converted to grayscale, resized to 28 × 28 and normalised by the same function the training data goes through ([preprocessing.py](src/fashion_mlops/preprocessing.py)), so inference input always matches training input.

| Status | When |
|---|---|
| `200` | Prediction returned |
| `400` | The file is empty or cannot be decoded as an image |
| `413` | The file is larger than 5 MB |
| `415` | The upload is not an image type |
| `422` | The `file` field is missing |
| `503` | No model is available yet, or MinIO cannot be reached |

The classes are: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

## Streamlit UI

Open <http://localhost:8501>.

1. Upload one image. Ten test-set images, one per class, are in [samples/](samples/).
2. Check the preview and press **Predict**.
3. Read the predicted class, the confidence and the probability of every class.

The sidebar shows whether the API is reachable and which model it is serving. The UI never runs the model itself; every prediction is a request to the FastAPI service.

Fashion-MNIST images show a light item on a black background, so the model works best on images in that style. An ordinary product photo on a white background is outside what it was trained on and may be misclassified.

## Running without Docker

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). MinIO and MLflow still run in Compose; the Python parts run on the host and read `.env`.

```bash
uv sync --all-extras
docker compose up -d minio mlflow webhook-receiver

uv run fashion-train --config configs/train.toml
uv run uvicorn fashion_mlops.api.main:app --port 8000
uv run streamlit run src/fashion_mlops/ui/app.py
```

To regenerate the sample images: `uv run python scripts/export_samples.py`.

## Tests

```bash
uv sync --all-extras
uv run pytest
```

The tests need no running services; MinIO, MLflow and the network are replaced by in-memory fakes.

| File | Covers |
|---|---|
| `tests/test_config.py` | Defaults, TOML loading, override precedence, rejection of invalid values |
| `tests/test_model_and_preprocessing.py` | Model output shape, normalisation, inference preprocessing equals training preprocessing, reproducible split |
| `tests/test_storage.py` | Versioned keys, no overwriting, latest pointer, pinned model, missing object and unavailable storage errors |
| `tests/test_webhook.py` | Payload contents, and that no webhook is sent when training or the upload fails |
| `tests/test_api.py` | Prediction response, input validation (400/413/415/422), 503 without a model, model hot-reload |
| `tests/test_ui.py` | UI messages for an unreachable API and a missing model, and display of a prediction |

## Project layout

```text
├── pyproject.toml              project metadata and dependencies (one extra per service)
├── uv.lock                     pinned dependency versions
├── .env.example                environment variables with placeholder values
├── docker-compose.yml          minio, mlflow, webhook-receiver, trainer, api, ui
├── docker/
│   ├── Dockerfile.app          image for the trainer and the API
│   └── Dockerfile.ui           image for the Streamlit UI (no PyTorch)
├── configs/train.toml          example training configuration
├── src/fashion_mlops/
│   ├── config.py               TrainConfig and ServiceSettings
│   ├── model.py                LeNet5, class names, model (de)serialisation
│   ├── data.py                 dataset download and seeded train/validation split
│   ├── preprocessing.py        the transform shared by training and inference
│   ├── train.py                training loop and the train → MLflow → MinIO → webhook flow
│   ├── storage.py              MinIO upload, download and model selection
│   ├── webhook.py              payload and delivery
│   ├── api/main.py             FastAPI service
│   └── ui/app.py               Streamlit app
├── scripts/
│   ├── webhook_receiver.py     local webhook receiver
│   └── export_samples.py       writes the sample images
├── samples/                    ten test images and a sample webhook payload
└── tests/
```
