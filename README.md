# MLOps projects

Assignments for the course *MLOps for Predictive Modeling* (McGill University).

| Project | Description |
|---|---|
| [assignment-1-fashion-mnist](assignment-1-fashion-mnist/) | Train, track, store and serve a Fashion-MNIST classifier: LeNet-5 in PyTorch, MLflow tracking, MinIO model storage, a training-complete webhook, and a FastAPI + Streamlit prediction stack, all run with Docker Compose |

Each project is self-contained. Open its folder and follow its README; for example:

```bash
cd assignment-1-fashion-mnist
cp .env.example .env
docker compose up -d minio mlflow webhook-receiver
docker compose run --rm trainer
docker compose up -d api ui
```
