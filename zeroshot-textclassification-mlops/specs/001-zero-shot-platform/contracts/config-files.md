# Contract: Configuration

## Environment variables

Values come from the process environment, or from a local `.env` file that Git ignores.

| Variable | Default | Purpose |
|---|---|---|
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO address. Compose sets `minio:9000` for the optional app container. |
| `MINIO_ACCESS_KEY` | None, required | MinIO access key. |
| `MINIO_SECRET_KEY` | None, required | MinIO secret key. |
| `MINIO_SECURE` | `false` | Use HTTPS when `true`. |
| `MINIO_BUCKET` | `zeroshot-learning-output` | Bucket for prediction records. |
| `MINIO_IMAGE` | `bitnamilegacy/minio:2025.7.23-debian-12-r5` | Compose image for local MinIO. |
| `MODEL_DIR` | `./models` | Controlled model directory. |
| `HF_HOME` | `./.cache/huggingface` | Hugging Face cache root. |
| `HF_TOKEN` | Empty | Optional Hugging Face token. |
| `MODELS_CONFIG` | `configs/models.yaml` | Model configuration file. |
| `LABELS_CONFIG` | `configs/labels.yaml` | Label configuration file. |
| `MAX_INPUT_CHARS` | `10000` | Maximum input text length. |
| `APP_VERSION` | `0.1.0` | Version stored with each prediction. |
| `LOG_LEVEL` | `INFO` | Logging level. |

`.env.example` holds placeholder values only. `minioadmin` is acceptable there as a documented local-only
default. Missing required values produce a clear error that names the variable but never shows its value.

## `configs/models.yaml`

```yaml
models:
  - id: MoritzLaurer/deberta-v3-base-zeroshot-v2.0
    name: DeBERTa Zero-Shot v2
    task: zero-shot-classification
    revision: 8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8

  - id: facebook/bart-large-mnli
    name: BART MNLI
    task: zero-shot-classification
    revision: d7645e127eaf1aefc7862fd59a17a5aa8558b8ce
```

## `configs/labels.yaml`

```yaml
default: general-topics
label_sets:
  - name: general-topics
    labels:
      - sports
      - politics
      - technology
      - finance
      - health
      - education
```

The admin page rewrites this file when label sets change. Validation rules are in
[data-model.md](../data-model.md).
