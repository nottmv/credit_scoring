#!/usr/bin/env bash
# Локальный стек без Docker: MLflow :5000 + FastAPI :8000
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p reports mlruns

pkill -f "uvicorn src.api.main:app" 2>/dev/null || true
pkill -f "gunicorn.*mlflow" 2>/dev/null || true
pkill -f "mlflow server" 2>/dev/null || true
sleep 1

export PYTHONPATH="$ROOT"
export MODEL_PATH="${MODEL_PATH:-$ROOT/models/model_bundle_catboost.pkl}"
export EVENTS_JSONL_PATH="$ROOT/reports/events.jsonl"
export DRIFT_REPORT_PATH="$ROOT/reports/last_drift.json"
export CHAMPION_METRICS_PATH="$ROOT/reports/champion_metrics.json"
export ADMIN_RELOAD_TOKEN="${ADMIN_RELOAD_TOKEN:-dev-token-change-me}"

mkdir -p "$ROOT/mlruns/artifacts"
# sqlite URI относительно каталога mlruns (без абсолютного пути в URI)
echo "Starting MLflow on http://127.0.0.1:5000 ..."
(
  cd "$ROOT/mlruns"
  nohup mlflow server --host 127.0.0.1 --port 5000 \
    --backend-store-uri "sqlite:///mlflow.db" \
    --default-artifact-root "mlflow-artifacts:/" \
    --artifacts-destination "${ROOT}/mlruns/artifacts" \
    --serve-artifacts \
    > /tmp/credit_scoring_mlflow.log 2>&1 &
)
sleep 2

echo "Starting API on http://127.0.0.1:8000 ..."
PY="${PYTHON:-python3}"
if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; fi
nohup "$PY" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 \
  > /tmp/credit_scoring_api.log 2>&1 &
sleep 2

for i in 1 2 3 4 5 6 7 8 9 10; do
  if curl -sf http://127.0.0.1:8000/health >/dev/null; then
    echo "OK: API http://127.0.0.1:8000/docs | dashboard http://127.0.0.1:8000/dashboard | metrics http://127.0.0.1:8000/metrics"
    echo "OK: MLflow http://127.0.0.1:5000"
    echo "Logs: /tmp/credit_scoring_api.log /tmp/credit_scoring_mlflow.log"
    exit 0
  fi
  sleep 1
done
echo "API did not become healthy in time. See /tmp/credit_scoring_api.log"
tail -30 /tmp/credit_scoring_api.log
exit 1
