# Image shared by the trainer and the FastAPI prediction service (both need PyTorch).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# UV_NO_CACHE keeps uv's download cache out of the image layers.
ENV UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first: this layer is rebuilt only when pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --extra train --extra api

COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev --extra train --extra api

# Run as an unprivileged user; /app/data is where the trainer downloads Fashion-MNIST.
RUN useradd --create-home app && mkdir -p /app/data && chown -R app:app /app/data
USER app

EXPOSE 8000
CMD ["uvicorn", "fashion_mlops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
