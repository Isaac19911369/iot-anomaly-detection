# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only PyTorch keeps the image small; swap the index URL for a CUDA build if needed.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install ".[serve,tracking]"

RUN useradd --create-home --uid 1000 app && mkdir -p /app/artifacts /app/data \
    && chown -R app /app
USER app

ENV MODEL_DIR=/app/artifacts/model
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "iot_anomaly.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
