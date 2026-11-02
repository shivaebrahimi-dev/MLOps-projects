.PHONY: install minio-up minio-down app lint format test test-integration

install:
	pip install -r requirements.txt

minio-up:
	docker compose up -d --wait minio minio-console

minio-down:
	docker compose down

app:
	streamlit run app/streamlit_app.py

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .

test:
	pytest -m "not integration"

test-integration:
	pytest -m integration
