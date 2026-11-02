# Research: Zero-Shot Text Classification Platform

Findings were checked on 2026-11-01. "Tested" means the check was run in an isolated environment.
"Not yet tested" items become validation tasks during implementation.

## R1. Runtime and dependency versions

- **Decision**: Python 3.12 with torch 2.15.0, transformers 5.22.1, huggingface_hub 2.4.0,
  streamlit 1.66.1, minio 7.2.20, PyYAML 6.0.3, python-dotenv 1.2.4, pytest 9.2.0, and ruff 0.17.3.
- **Rationale**: These are the current PyPI releases. Tested: they install together on Python 3.12
  on Windows. Python 3.12 is installed on the development machine and meets every package's minimum.
- **Alternatives considered**: Transformers 4.x, rejected because a new project has no reason to start
  on an older major version.

## R2. MinIO container image

- **Decision**: Use `bitnamilegacy/minio:2025.7.23-debian-12-r5` for the server and
  `bitnamilegacy/minio-object-browser:2.0.2-debian-12-r4` for the console, configurable through
  `MINIO_IMAGE` and `MINIO_CONSOLE_IMAGE`, as recorded in constitution decision DR-003.
- **Rationale**: `docker manifest inspect` could not find `minio/minio:latest` on Docker Hub or
  `quay.io/minio/minio:latest`, and the legacy download archive returned 410. MinIO's repository README
  says the community edition is now source-only.
- **Tested (T003)**: The server starts healthy with data at `/bitnami/minio/data`, stores and returns an
  object, and keeps it after a restart. Its digest is
  `sha256:6dabb4a2088c9a79908de3bc05f4586c23ad2182c8908e7e3acbf61c1467fb20`.
- **Console**: The server image passes `--console-address` but nothing listens on that port, so the demo's
  console needs the separate object-browser image. That image must be started with the `server` command.
  Tested: its page loads on port 9001, a wrong password returns 401, the right one succeeds, and it lists the
  bucket.
- **Alternatives considered**:
  - `pgsty/minio:RELEASE.2026-08-04T00-00-00Z`: newer and follows the official image's layout, but it is
    a third-party community build. It is the first fallback if the Bitnami image fails.
  - Building from source: closest to MinIO's guidance, but it adds a slow, fragile setup step.
  - An authenticated pull: rejected because each new developer and CI would need registry credentials.

## R3. Bucket name

- **Decision**: Default to `zeroshot-learning-output` (DR-001).
- **Rationale**: Tested on running MinIO: asked directly about `zeroshot_learning_output`, the server returns
  the error `InvalidBucketName`. `make_bucket` in the Python client raises `ValueError` for that name, and
  `mc mb` fails with "Bucket name contains invalid characters". All accept `zeroshot-learning-output`. An
  integration test keeps checking that the application reports this clearly.

## R4. Model download and readiness

- **Decision**: Pin each model's revision in `configs/models.yaml`. Download with
  `huggingface_hub.snapshot_download` into a temporary folder under the model directory, write a
  completion marker, then rename the folder into its final location. A model is ready only if its final
  folder contains the marker.
- **Rationale**: A rename within one file system is effectively atomic, so interrupted downloads cannot
  look ready. This satisfies FR-009 and FR-010. Tested: `snapshot_download` accepts `revision`,
  `local_dir`, and `ignore_patterns`.
- **Pinned revisions**:
  - `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`: `8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8`
  - `facebook/bart-large-mnli`: `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce`
- **File filter**: Download only configuration, tokenizer, and `model.safetensors` files. The repositories
  also hold ONNX, Flax, Rust, `pytorch_model.bin`, and `training_args.bin` files that are not needed. This
  saves disk space and avoids pickle-format files.
- **Alternatives considered**: Loading directly from the Hugging Face cache, rejected because the
  assignment requires a controlled project directory and explicit readiness status.

## R5. Hugging Face cache behavior

- **Decision**: Use `HF_HOME` only, and drop `TRANSFORMERS_CACHE` from the assignment's example. Cache
  cleanup removes only the `hub` and `xet` folders under `HF_HOME`.
- **Rationale**: The Transformers cache code (`utils/hub.py`) on its main branch recommends `HF_HOME`
  and no longer refers to `TRANSFORMERS_CACHE`. `HF_HOME` can also hold a saved login token, so it must
  not be deleted as a whole.
- **Consequence**: Downloads to a `local_dir` bypass most of the shared cache, so cleanup may report little
  to remove. The README must explain this so the demo's result is not surprising.
- **Alternatives considered**: Downloading through the cache and copying to the project directory, rejected
  because it stores each large model twice.

## R6. Tokenizer dependencies

- **Decision**: Do not add `sentencepiece` for now.
- **Rationale**: Tested without it: the DeBERTa and BART tokenizers both load at their pinned revisions and
  tokenize text.
- **Not yet tested**: Full inference with the real models. If it needs `sentencepiece`, DR-002 already allows
  tokenizer dependencies.

## R7. PyTorch backend enforcement

- **Decision**: Load the pipeline with the local model path and confirm that `pipeline.model` is a
  `torch.nn.Module`, raising an error otherwise.
- **Rationale**: Tested: `pipeline()` in Transformers 5.22.1 has no `framework` parameter, so backend choice
  cannot be passed as an argument, and an explicit check meets the constitution's rule against silently
  switching backends. A tiny test model also ran through the zero-shot pipeline, returning ranked labels
  with scores summing to 1.0 in single-label mode.

## R8. Streamlit and service boundaries

- **Decision**: Pages create services through one shared `app/ui_state.py` helper that uses
  `st.cache_resource`. Services never import Streamlit. The classifier keeps at most one loaded pipeline and
  replaces it when a different model is selected.
- **Rationale**: This satisfies the constitution's separation rule and NFR-005.

## R9. CI location

- **Decision**: Put the workflow at the repository root, in `.github/workflows/zeroshot-ci.yml`, filtered to
  `zeroshot-textclassification-mlops/**` and running from that folder.
- **Rationale**: GitHub only runs workflows from the repository root, and this project is in a subfolder of a
  repository shared with another assignment. The assignment's example places it inside the project.
- **Consequence**: CI uses CPU-only PyTorch for speed and runs MinIO as a service container.

## R10. Optional features

- **Decision**: Include inference-latency measurement and a Dockerfile with an opt-in Compose profile. Skip
  the status panel.
- **Rationale**: Latency is cheap and already has a place in the record. The assignment's suggested structure
  includes a Dockerfile and asks for Docker support. A status panel adds work without helping any acceptance
  criterion.
