# Fashion-MNIST MLOps Pipeline (Homework 1)

This note summarises the first course project: a pipeline that trains, tracks, stores and serves an image
classifier for the Fashion-MNIST dataset. Every service runs with Docker Compose.

## Model and training

The model is a LeNet-5-style convolutional network adapted to 28 × 28 grayscale images. It has two 5 × 5
convolution layers with 6 and 16 channels, each followed by ReLU and 2 × 2 max pooling, and then three fully
connected layers with 120, 84 and 10 units. With the default configuration the model reaches about 90% validation
accuracy after five epochs, and training takes well under a minute on a CPU.

The default training settings are: 5 epochs, batch size 64, learning rate 0.001, the Adam optimizer, cross-entropy
loss, seed 42, and a validation split of 0.1 (10% of the 60,000 training images are held out). The optimizer can also
be SGD with momentum 0.9. Early stopping is optional and is disabled when its patience is 0.

Settings are resolved in this order, highest priority first: a command-line flag, an environment variable with the
`TRAIN_` prefix, the TOML file passed with `--config`, and finally the built-in default. Invalid values and unknown
keys are rejected before training starts.

## Order of a training run

A run performs these steps in order and stops at the first failure:

1. Check that MLflow and MinIO are reachable, and create the bucket if needed.
2. Train, logging metrics to MLflow after each epoch.
3. Upload the model to MinIO.
4. Record the MinIO reference in MLflow.
5. Send the training-complete webhook.

If any step fails, the trainer exits with a non-zero status and the MLflow run is marked FAILED.

## Experiment tracking with MLflow

MLflow records every training setting as a parameter. After each epoch it logs `train_loss`, `val_loss` and
`val_accuracy`. At the end it logs `final_val_accuracy`, `final_val_loss` and `epochs_run`. It also stores two
artifacts: `config.json` with the resolved configuration, and `model_info.json` with the bucket, object key, size,
SHA-256 hash and class names. MLflow keeps a reference to the model, not a second copy.

## Model storage in MinIO

Trained models are stored in the MinIO bucket `fashion-minst-bucket` (the spelling "minst" is the exact name the
assignment required). Each model is saved as `model/<mlflow-run-id>/model.pt`, so a later run never overwrites an
earlier one and every model can be traced back to its MLflow run. The file `model/latest.json` points to the most
recently uploaded model. The model file is loaded with `weights_only=True`, so loading it cannot execute code.

## Webhook

After a run has trained, logged to MLflow and uploaded its model, the trainer sends a JSON `training_complete`
notification to the configured webhook URL. If training, logging or the upload fails, no notification is sent. The
payload contains the run id, the experiment name, the model's bucket, object key and `s3://` URI, and the final
validation metrics.

## Serving predictions

A FastAPI backend serves single-image predictions at `/predict`, and a Streamlit UI on port 8501 lets a user upload
an image. By default the API serves the newest model by reading `model/latest.json` on each request, so a new
training run is picked up without a restart. A specific model can be pinned with `MODEL_OBJECT_KEY`. If the model,
the bucket or MinIO itself is unavailable, the API answers with HTTP 503 and says which problem occurred.

| Service | Port | Purpose |
|---|---|---|
| Streamlit UI | 8501 | Prediction interface |
| FastAPI | 8000 | Prediction backend |
| MLflow | 5000 | Experiment tracking |
| MinIO | 9000 (API), 9001 (console) | Model storage |
| Webhook receiver | 8090 | Prints received webhooks for testing |
