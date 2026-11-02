FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# The CPU build of PyTorch is much smaller than the default Linux build.
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY app ./app
COPY configs ./configs
COPY .streamlit ./.streamlit
RUN mkdir -p models .cache && useradd --create-home appuser && chown -R appuser /app
USER appuser

# Port publishing in docker-compose.yml limits access to this computer.
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health', timeout=2)"

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.headless=true"]
