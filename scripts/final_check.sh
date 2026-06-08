#!/usr/bin/env bash
# Финальная проверка перед показом / без Cursor.
# Запуск из корня репозитория: ./scripts/final_check.sh
#   --docker     дополнительно: docker compose up, curl сервисов (порты см. ниже)
#
# Переменные окружения:
#   SKIP_STACK=1       с --docker не поднимать compose (только config)
#   MLFLOW_HOST_PORT   по умолчанию 5000; если занят — 5001
#   API_HOST_PORT      по умолчанию 8000; если занят — 8002
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WITH_DOCKER=0
for arg in "$@"; do
  case "$arg" in
    --docker) WITH_DOCKER=1 ;;
  esac
done

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
fi

PY_MM="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PY_MM" != "3.11" ]]; then
  echo "WARN: для совпадения с Docker/CI лучше Python 3.11; сейчас ${PY_MM}. При ошибках pip — см. README (раздел про venv)."
fi

if ! command -v flake8 >/dev/null 2>&1 || ! command -v pytest >/dev/null 2>&1; then
  echo "FAIL: нет flake8 или pytest в текущем Python."
  echo "      Активируйте venv: source .venv/bin/activate"
  echo "      Затем: pip install -U pip setuptools wheel && pip install -r requirements.txt"
  exit 1
fi

echo "=== [1/4] Артефакт модели ==="
if [[ ! -f "$ROOT/models/model_bundle_catboost.pkl" ]]; then
  echo "FAIL: нет models/model_bundle_catboost.pkl"
  echo "      Выполните: make train   или   make fetch-data && make train"
  exit 1
fi
echo "OK: models/model_bundle_catboost.pkl"

echo "=== [2/4] Линтер и тесты ==="
flake8 --max-line-length=100 --extend-exclude=.dvc src tests scripts
PYTHONPATH="$ROOT" pytest -q tests
echo "OK: flake8 + pytest"

echo "=== [3/4] Docker ==="
docker version >/dev/null
docker compose version >/dev/null
docker compose config >/dev/null
echo "OK: docker / compose config"

if [[ "$WITH_DOCKER" -eq 0 ]]; then
  echo
  echo "Готово (без запуска контейнеров)."
  echo "Полный смоук со стеком:  ./scripts/final_check.sh --docker"
  echo "Или вручную:             make docker-up"
  exit 0
fi

if [[ "${SKIP_STACK:-}" == "1" ]]; then
  echo "SKIP_STACK=1 — пропуск подъёма compose."
  exit 0
fi

pick_port() {
  local default="$1" alt="$2"
  if lsof -nP -iTCP:"$default" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "$alt"
  else
    echo "$default"
  fi
}

API_HOST_PORT="${API_HOST_PORT:-$(pick_port 8000 8002)}"
MLFLOW_HOST_PORT="${MLFLOW_HOST_PORT:-$(pick_port 5000 5001)}"
PROMETHEUS_HOST_PORT="${PROMETHEUS_HOST_PORT:-$(pick_port 9090 9091)}"
GRAFANA_HOST_PORT="${GRAFANA_HOST_PORT:-$(pick_port 3000 3001)}"

export API_HOST_PORT MLFLOW_HOST_PORT PROMETHEUS_HOST_PORT GRAFANA_HOST_PORT

echo "=== [4/4] Docker Compose (порты хоста) ==="
echo "    API:$API_HOST_PORT  MLflow:$MLFLOW_HOST_PORT  Prometheus:$PROMETHEUS_HOST_PORT  Grafana:$GRAFANA_HOST_PORT"

# По умолчанию без --build (быстро; образы уже есть после первого make docker-up).
# Пересборка: FINAL_CHECK_COMPOSE_BUILD=1 ./scripts/final_check.sh --docker
if [[ "${FINAL_CHECK_COMPOSE_BUILD:-}" == "1" ]]; then
  docker compose up --build -d
else
  docker compose up -d
fi

echo "Ожидание health API..."
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${API_HOST_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

curl -sf "http://127.0.0.1:${API_HOST_PORT}/health" | python3 -m json.tool >/dev/null
echo "OK: GET http://127.0.0.1:${API_HOST_PORT}/health"

curl -sf "http://127.0.0.1:${API_HOST_PORT}/metrics" | head -1 >/dev/null
echo "OK: GET .../metrics"

curl -sf "http://127.0.0.1:${MLFLOW_HOST_PORT}/" >/dev/null
echo "OK: MLflow UI http://127.0.0.1:${MLFLOW_HOST_PORT}/"

curl -sf "http://127.0.0.1:${PROMETHEUS_HOST_PORT}/-/healthy" >/dev/null
echo "OK: Prometheus http://127.0.0.1:${PROMETHEUS_HOST_PORT}/"

curl -sf "http://127.0.0.1:${GRAFANA_HOST_PORT}/login" >/dev/null
echo "OK: Grafana http://127.0.0.1:${GRAFANA_HOST_PORT}/ (admin/admin)"

echo
echo "Все проверки прошли. Контейнеры работают."
echo "Остановка: make docker-down"
