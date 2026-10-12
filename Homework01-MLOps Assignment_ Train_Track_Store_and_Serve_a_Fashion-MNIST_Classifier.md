

# MLOps Assignment: Train, Track, Store, and Serve a Fashion-MNIST Classifier

## Objective

Build an end-to-end MLOps application that trains a LeNet-5-style model on Fashion-MNIST, tracks experiments in MLflow, stores the trained model in MinIO, and serves single-image predictions through a FastAPI backend with a Streamlit user interface. Use **Docker Compose** to run the system’s services.

## Required workflow

```text
Configure training run
        ↓
Train LeNet-5-style model on Fashion-MNIST
        ↓
Log parameters, metrics, and model information to MLflow
        ↓
Upload model artifact to MinIO
        ↓
Send training-complete webhook after successful upload
        ↓
Run prediction stack with Docker Compose
        ↓
Upload an image in Streamlit → FastAPI inference → show predicted class
```

## Functional requirements

### 1. Project configuration and dependencies

- Use a TOML-based project file (for example, `pyproject.toml`) to declare dependencies and project metadata.
- Keep training settings configurable rather than hard-coded. Configuration may be supplied through a TOML file, environment variables, command-line arguments, or a documented combination.
- Provide a README with setup and run instructions.

### 2. Model training

- Train a LeNet-5-style convolutional neural network using Fashion-MNIST.
- The model must accept a grayscale **28 × 28** image and predict one of the ten Fashion-MNIST classes.
- Support a train/validation split and report validation performance.
- Make the principal training parameters configurable, including at least:
  - Number of epochs
  - Batch size
  - Learning rate
  - Optimizer
  - Loss function
  - Random seed
  - Validation split or validation dataset settings
- Suitable additional configurable parameters include optimizer-specific settings (such as momentum or weight decay), learning-rate scheduler, early stopping/patience, and data-loader settings. Any optional settings implemented should be documented.
- Record the actual configuration used for each run so that a training run can be reproduced.

### 3. MLflow experiment tracking

Log each training run to MLflow. At minimum, record:

- **Parameters:** epochs, batch size, learning rate, optimizer, loss function, random seed, and other configured training settings.
- **Metrics:** training and validation loss by epoch, and final validation accuracy. Additional metrics such as precision, recall, or a confusion matrix may be included.
- **Artifacts:** the trained model or a reference to its MinIO object, plus useful run metadata.

The README should explain how to view the experiment and its runs in the MLflow UI.

### 4. MinIO model storage

- Store the trained model in MinIO in bucket **`fashion-minst-bucket`**, under the **`/model/`** prefix. Use the requested bucket spelling exactly; document that `minst` is intentional if retained.
- A model object key could follow a pattern such as `model/<run-id>/model.pt` or `model/<run-id>/model.onnx`or any other options. Include a version or run identifier so later training runs do not silently overwrite earlier models.
- Make MinIO endpoint and credentials configurable via environment variables. Do not commit secrets to source control.
- Ensure the prediction service can retrieve the selected model from MinIO, and document how it identifies which model to load.
- Create the bucket/prefix as part of startup or setup if needed, and handle missing objects or unavailable storage with a clear error.

### 5. Training-complete webhook

- After training succeeds **and** the model has been uploaded successfully to MinIO, send a webhook notification.
- The payload should include useful information such as status, MLflow run ID, model bucket and object key, and validation accuracy.
- Do not send a success notification if training, MLflow logging (where required), or MinIO upload fails. Report failures clearly.
- Make the webhook URL configurable and document how to test notifications, for example with a local webhook receiver.

### 6. Docker Compose deployment

- Use Docker Compose to define and run the application services. The required deployment must include the prediction backend and Streamlit UI; include MinIO and MLflow services in Compose where appropriate for a reproducible local environment.
- Configure service dependencies, ports, persistent volumes for MinIO/MLflow data, and environment variables.
- Provide a documented startup sequence that ensures training has completed and the model is present in MinIO before prediction is used. Training may be run as a Compose service or one-shot command; prediction and UI services should be runnable with Compose.
- The UI should give a helpful message if the model is not yet available.

### 7. FastAPI prediction backend

- Implement an inference endpoint that accepts a **single image** (for example, multipart file upload), preprocesses it consistently with training, loads the selected model from MinIO, and returns the predicted class.
- Return a readable class name and, optionally, per-class probabilities or confidence scores.
- Validate image inputs and return useful client errors for unsupported or invalid files.
- Include a health-check endpoint and document the API endpoint and request format.

### 8. Streamlit interface

- Provide a simple, polished Streamlit interface on top of the FastAPI backend.
- Let the user upload one image, preview it, and request a prediction.
- Display the predicted Fashion-MNIST label and, if implemented, confidence/probability information.
- Show understandable messages for API or model-loading errors.
- The Streamlit app should call the FastAPI service rather than duplicating model inference logic.

## Deliverables

1. Source code for training, model definition, data handling, and inference.
2. TOML project/dependency configuration and example training configuration.
3. Dockerfile(s) and `docker-compose.yml`.
4. MLflow experiment with logged parameters, metrics, and model/run metadata.
5. A trained model stored in MinIO under `fashion-minst-bucket/model/`.
6. Training-complete webhook implementation and sample payload.
7. FastAPI inference API and Streamlit prediction interface.
8. README covering installation, configuration, training, MLflow, MinIO, webhook testing, Compose startup, and prediction usage.
9. Basic tests for configuration, preprocessing, API input validation, and/or prediction response.

## Acceptance criteria

- A user can configure and launch a training run without editing source code.
- The run records the configured hyperparameters and validation metrics in MLflow.
- A successful training run uploads a versioned model artifact to the specified MinIO bucket and prefix.
- The webhook is sent only after the artifact upload succeeds and contains the run/model references.
- Docker Compose starts the FastAPI prediction service and Streamlit UI, and the backend can load the model from MinIO.
- A user can upload a single Fashion-MNIST-format image in Streamlit and see the predicted class.
- Setup and run steps are documented and do not require secrets to be embedded in the repository.

## Suggested implementation note

Keep training and inference as separate processes/services. Training produces a versioned model artifact in MinIO; inference retrieves a selected artifact at startup or on demand. Streamlit acts as the presentation layer and sends uploaded images to FastAPI. This makes the training-complete webhook a clear lifecycle notification without coupling it to the prediction UI.