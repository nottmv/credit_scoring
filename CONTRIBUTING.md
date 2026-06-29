# Участие в проекте и MLOps-практики

## Git: feature-branch flow

- **`main`** — стабильная ветка, деплой только через MR/PR.
- **`feature/<кратко>`** — одна задача на ветку.
- **`fix/<кратко>`** — исправления.

Слияние в `main` после ревью и зелёного CI.

## Conventional Commits

Формат: `<type>(scope): описание`

Типы: `feat`, `fix`, `docs`, `refactor`, `test`, `ci`, `chore`, `build`.

Примеры: `feat(api): batch score`, `fix(drift): PSI bins`, `ci: add retrain workflow`.

## Данные и DVC

Артефакты в общей папке Google Drive:  
[mlops — credit_scoring.csv и модели](https://drive.google.com/drive/folders/1HbYd0bgGCuGbBdKDmH0dCGJQt-pOHcGv?usp=drive_link)

**Вариант A — DVC (рекомендуется для воспроизводимости):**

```bash
pip install 'dvc[gdrive]'
# remote уже настроен: gdrive://<folder_id>
dvc pull
```

При первом `dvc pull` откроется OAuth в браузере (Google).

**Вариант B — прямое скачивание папки (gdown):**

```bash
pip install gdown
make fetch-data
# или: python scripts/download_drive_artifacts.py
```

## Локальный запуск без Docker

```bash
make requirements
make local-up
```

Если нет `data/raw/credit_scoring.csv`, обучение пойдёт на `synthetic_min.csv`; для полного датасета: `make fetch-data`.

**MLflow пустой**, пока не залогирован хотя бы один run: сначала `make local-up` (MLflow на :5000), затем **`make train-mlflow`** (или `python src/models/train_model.py ... --mlflow-uri http://127.0.0.1:5000`). Без `--mlflow-uri` / `MLFLOW_TRACKING_URI` обучение не пишет в UI.

## Docker на macOS

### Установка и `command not found: docker`

1. Установите [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/) (Apple Silicon или Intel — свой dmg).
2. Откройте **Docker Desktop** из Applications и дождитесь **Docker is running**.
3. В новом терминале проверьте:
   ```bash
   docker version
   docker compose version
   ```
4. Если команды всё ещё не находятся:
   - **Docker Desktop → Settings → Advanced**: включите **Docker CLI tools** / проверьте установку symlinks.
   - Добавьте в `PATH` (часто уже есть): `/usr/local/bin` или `~/.docker/bin` (см. настройки Docker Desktop).
5. Альтернатива без Docker Desktop — **Colima** + CLI (для продвинутых):
   ```bash
   brew install colima docker docker-compose
   colima start
   docker context use colima
   ```

### Запуск проекта в Docker

Нужен файл **`models/model_bundle_catboost.pkl`** (иначе API не пройдёт healthcheck):

```bash
make train          # или make fetch-data && make train
make docker-up      # docker compose up --build -d
make docker-ps      # статус контейнеров
```

Сервисы: **API** http://localhost:8000 , **MLflow** :5000 , **Prometheus** :9090 , **Grafana** :3000 (логин/пароль `admin`/`admin` из `docker-compose.yml`).

Остановка: **`make docker-down`**.

Prometheus ждёт готовности API (`depends_on` + `condition: service_healthy`), чтобы сразу скрейпить `/metrics`.

---

## Kubernetes (minikube)

### Установка инструментов

```bash
brew install minikube kubectl
# Docker должен быть доступен (Docker Desktop или Colima)
```

### Один скрипт (рекомендуется)

Из корня репозитория (модель `models/model_bundle_catboost.pkl` должна существовать):

```bash
make minikube-deploy
```

Скрипт `scripts/k8s_minikube.sh` делает: `minikube start` при необходимости → `eval "$(minikube docker-env)"` → сборка образов **внутри** Docker minikube → `minikube cp` модели в **`/tmp/credit-models`** на ноде → `kubectl apply` для `mlflow`, API, Service, **Prometheus**.

### Проверка и URL

```bash
kubectl get pods
kubectl get svc
minikube service credit-scoring-api --url
minikube service mlflow --url
minikube service prometheus --url
```

NodePort по умолчанию: API **30800**, MLflow **30500**, Prometheus **30909** (см. YAML).

### Вручную (как раньше)

```bash
minikube start
eval "$(minikube docker-env)"
make k8s-images
minikube ssh "sudo mkdir -p /tmp/credit-models && sudo chmod a+rwx /tmp/credit-models"
minikube cp ./models/model_bundle_catboost.pkl minikube:/tmp/credit-models/model_bundle_catboost.pkl
kubectl apply -f k8s/mlflow.yaml
kubectl apply -f k8s/local/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/prometheus.yaml
```

### Grafana в k8s

В манифестах репозитория Grafana **не** развёрнута в minikube (чтобы не дублировать тяжёлую конфигурацию). Для дашбордов удобно:

- пользоваться **Grafana из `docker compose`** на машине и временно добавить datasource Prometheus с URL из `minikube service prometheus --url`, **или**
- открывать только **Prometheus** (`/graph` на NodePort 30909).

### Job переобучения на кластере

Одноразово, после подготовки данных на ноде в `/tmp/credit-train/`:

```bash
kubectl apply -f k8s/manual/job-retrain.yaml
```

См. комментарии внутри файла.

## Цикл: дрифт → переобучение → reload

1. Положить новый батч, например `data/incoming/current.csv`.
2. `python scripts/drift_check.py --current data/incoming/current.csv --model-path models/model_bundle_catboost.pkl --fail-on-drift`
3. При деградации: `python scripts/retrain_pipeline.py --data data/raw/credit_scoring.csv --drift-current data/incoming/current.csv`
4. Обновить сервис: `POST /internal/reload-model` с заголовком `X-Admin-Token` (= `ADMIN_RELOAD_TOKEN` в compose).

Мониторинг: `/dashboard`, `/metrics`, Grafana.
