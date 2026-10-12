# Fashion-MNIST MLOps: train, track, store and serve

An end-to-end MLOps application that trains a LeNet-5-style model on Fashion-MNIST, tracks experiments in MLflow, stores the trained model in MinIO, sends a training-complete webhook, and serves predictions through a FastAPI backend with a Streamlit UI. Everything runs with Docker Compose.

> Work in progress: this README is completed as each part of the system is built.

## Local development setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
uv run pytest
```
