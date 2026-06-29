FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model_bundle_catboost.pkl \
    EVENTS_JSONL_PATH=/app/reports/events.jsonl \
    DRIFT_REPORT_PATH=/app/reports/last_drift.json \
    CHAMPION_METRICS_PATH=/app/reports/champion_metrics.json

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY setup.py requirements-docker.txt ./
COPY src ./src
RUN pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir -r requirements-docker.txt

RUN mkdir -p /app/models /app/reports

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
