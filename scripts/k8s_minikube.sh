#!/usr/bin/env bash
# Сборка образов в Docker minikube, копирование модели, apply манифестов.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

need() { command -v "$1" >/dev/null 2>&1 || { echo "Нет команды: $1 — см. CONTRIBUTING.md (Docker / minikube)"; exit 1; }; }
need docker
need kubectl
need minikube

minikube status >/dev/null 2>&1 || minikube start

echo ">>> Docker daemon minikube"
eval "$(minikube docker-env)"

echo ">>> Сборка образов"
docker build -t credit-scoring-api:latest .
docker build -f docker/mlflow/Dockerfile -t credit-scoring-mlflow:latest .

if [ ! -f models/model_bundle_catboost.pkl ]; then
  echo "Нет models/model_bundle_catboost.pkl — выполните: make train   или   make fetch-data && make train"
  exit 1
fi

echo ">>> Копирование модели на ноду minikube (/tmp/credit-models)"
minikube ssh "sudo mkdir -p /tmp/credit-models && sudo chmod a+rwx /tmp/credit-models" || true
minikube cp "${ROOT}/models/model_bundle_catboost.pkl" "minikube:/tmp/credit-models/model_bundle_catboost.pkl"

echo ">>> kubectl apply"
kubectl apply -f "${ROOT}/k8s/mlflow.yaml"
kubectl apply -f "${ROOT}/k8s/local/deployment.yaml"
kubectl apply -f "${ROOT}/k8s/service.yaml"
kubectl apply -f "${ROOT}/k8s/prometheus.yaml"
kubectl apply -f "${ROOT}/k8s/ingress.yaml" 2>/dev/null || echo "Ingress skipped (enable nginx ingress addon)"

echo ""
echo "Дождитесь Ready: kubectl get pods -w"
echo "URL (после Ready):"
echo "  API:       $(minikube service credit-scoring-api --url 2>/dev/null || echo 'minikube service credit-scoring-api --url')"
echo "  MLflow:    $(minikube service mlflow --url 2>/dev/null || echo 'minikube service mlflow --url')"
echo "  Prometheus: $(minikube service prometheus --url 2>/dev/null || echo 'minikube service prometheus --url')"
echo ""
echo "Grafana в minikube не развёрнута — используйте docker compose (порт 3000) и временно укажите Prometheus URL из команды выше."
