# Kubernetes (minikube)

| Файл | Назначение |
|------|------------|
| `mlflow.yaml` | PVC + Deployment + Service MLflow (NodePort 30500) |
| `deployment.yaml` | API; модель с **hostPath** `/tmp/credit-models` на ноде |
| `service.yaml` | API NodePort **30800** |
| `prometheus.yaml` | ConfigMap + Prometheus, скрейп `credit-scoring-api:8000`, NodePort **30909** |
| `manual/job-retrain.yaml` | Job переобучения (подключать вручную) |

Быстрый деплой из корня репозитория: **`make minikube-deploy`** (скрипт `scripts/k8s_minikube.sh`).
