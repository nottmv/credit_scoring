# Kubernetes (minikube)

| Файл | Назначение |
|------|------------|
| `mlflow.yaml` | PVC + Deployment + Service MLflow (NodePort 30500) |
| `local/deployment.yaml` | API local mode; модель с **hostPath** `/tmp/credit-models` на ноде |
| `prod/deployment.yaml` | API prod mode; образ из GHCR, модель из MLflow Registry |
| `service.yaml` | API NodePort **30800** |
| `prometheus.yaml` | ConfigMap + Prometheus, скрейп `credit-scoring-api:8000`, NodePort **30909** |
| `manual/job-retrain.yaml` | Job переобучения (подключать вручную) |

Быстрый деплой из корня репозитория: **`make minikube-deploy`** (скрипт `scripts/k8s_minikube.sh`).
