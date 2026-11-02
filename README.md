# MLOps projects

Assignments for the course *MLOps for Predictive Modeling* (McGill University).

| Project | Description |
|---|---|
| [fashion-mnist-mlops](fashion-mnist-mlops/) | Train, track, store and serve a Fashion-MNIST classifier: LeNet-5 in PyTorch, MLflow tracking, MinIO model storage, a training-complete webhook, and a FastAPI + Streamlit prediction stack, all run with Docker Compose |
| [zeroshot-textclassification-mlops](zeroshot-textclassification-mlops/) | Classify text into labels chosen at request time, without training a model: Hugging Face zero-shot models, Streamlit admin and classification pages, and JSON results stored in MinIO, built spec-first with Spec Kit |

Each project is self-contained. Open its folder and follow its README; for example:

```bash
cd fashion-mnist-mlops
cp .env.example .env
docker compose up -d minio mlflow webhook-receiver
docker compose run --rm trainer
docker compose up -d api ui
```

```bash
cd zeroshot-textclassification-mlops
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d --wait minio minio-console
streamlit run app/streamlit_app.py
```
