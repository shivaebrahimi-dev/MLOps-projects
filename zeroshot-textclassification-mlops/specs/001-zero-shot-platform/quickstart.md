# Quickstart Validation: Zero-Shot Text Classification Platform

This guide proves that the feature works end to end once it is implemented. The README remains the full setup
guide. Commands use PowerShell and run from `zeroshot-textclassification-mlops/`.

## Prerequisites

- Python 3.12
- Docker Desktop, running
- Internet access and enough disk space for the models

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
docker compose up -d minio
streamlit run app/streamlit_app.py
```

**Expected**: MinIO's console opens at `http://localhost:9001`, and the app opens at `http://localhost:8501`.

## Scenario 1: Manage models (Story 2)

1. Open the admin page. **Expected**: Both configured models are listed as not downloaded.
2. Download the DeBERTa model. **Expected**: It becomes ready and shows its revision and path.
3. Download it again. **Expected**: It reports that it is already available, and no files transfer.
4. Download the BART model. **Expected**: It becomes ready too.

## Scenario 2: Manage labels (Story 3)

1. Try to save a set with one label. **Expected**: It is refused with a clear message.
2. Add a duplicate label such as `Sports` to a set with `sports`. **Expected**: It is rejected.
3. Save a set of `sports`, `politics`, `technology`, and `finance`, then restart Streamlit. **Expected**: The set is
   still available.

## Scenario 3: Classify and store (Story 1)

1. On the classification page, select DeBERTa and the saved set, then submit
   `The team won the championship after a very close game.`
   **Expected**: `sports` is predicted, all labels are ranked, and the model, timestamp, and storage path appear.
2. In the MinIO console, open `zeroshot-learning-output`, then the shown path under `predictions/`.
   **Expected**: The JSON object matches [prediction-record.schema.json](contracts/prediction-record.schema.json).
3. Submit empty text. **Expected**: It is rejected, and no new object appears.

## Scenario 4: Handle failures

1. Run `docker compose stop minio`, then classify text. **Expected**: The result appears with a clear "not saved" error.
2. Run `docker compose start minio`, then classify again. **Expected**: The new result is stored.

## Scenario 5: Delete and clean up (Story 4)

1. Delete the BART model after confirming. **Expected**: It becomes not downloaded and leaves the classification page.
2. Clear the cache after confirming. **Expected**: The outcome is reported, and the ready DeBERTa model still works.

## Scenario 6: Run the quality checks (Story 5)

```powershell
ruff check .
ruff format --check .
pytest -m "not integration"
pytest -m integration
```

**Expected**: All checks pass. The integration tests need MinIO to be running.

## Scenario 7: Check the repository

```powershell
git ls-files | Select-String -Pattern '(^|/)(\.env$|models/.+\.safetensors|\.cache/)'
```

**Expected**: No output.

## Shut down

```powershell
docker compose down
```

Stored predictions remain in the Docker volume. `docker compose down -v` deletes them too.
