# Kubernetes prod (Argo CD sync path: k8s/prod)

Argo CD Application `credit-scoring` синхронизирует **всю папку** `k8s/prod/`:

| Файл | Назначение |
|------|------------|
| `deployment.yaml` | API из GHCR, initContainer wait-mlflow, baked model fallback |
| `service.yaml` | NodePort **30800** |
| `mlflow.yaml` | MLflow + PVC + ClusterIP Service `mlflow:5000` |

## Перед первым деплоем

### 1. GHCR pull secret (ImagePullBackOff без него)

```bash
export GHCR_USER=nottmv
export GHCR_TOKEN=<PAT с read:packages>
chmod +x scripts/bootstrap-ghcr-secret.sh
./scripts/bootstrap-ghcr-secret.sh
```

Убедитесь, что пакет `ghcr.io/nottmv/credit-scoring-api` **существует** (CI push на master) и доступен токену.

Проверка образа:
```bash
docker pull ghcr.io/nottmv/credit-scoring-api:latest
```

### 2. Argo CD sync

```bash
kubectl apply -f argocd/application.yaml
argocd app sync credit-scoring
kubectl get pods -w
kubectl describe pod -l app=credit-scoring-api   # Events → ImagePullBackOff?
kubectl logs -l app=credit-scoring-api --previous   # CrashLoopBackOff
```

### 3. Типичные причины Degraded

| Симптом | Причина | Решение |
|---------|---------|---------|
| **ImagePullBackOff** | Нет образа с тегом SHA / нет `ghcr-secret` | CI green + bootstrap-ghcr-secret |
| **CrashLoopBackOff** | Нет модели + MLflow недоступен | Образ теперь содержит `.pkl`; MLflow в `k8s/prod/mlflow.yaml` |
| Много ReplicaSet | Неудачные rollout | `revisionHistoryLimit: 3`, `maxUnavailable: 0` |

### 4. Регистрация модели в MLflow (опционально)

После первого старта API работает на baked model. Для Registry:

```bash
kubectl apply -f k8s/manual/job-retrain.yaml
```
