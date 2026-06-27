# Credit Scoring — MLOps система

Полный цикл MLOps: **новые данные → обнаружение деградации → переобучение → обновление модели в сервисе → мониторинг через веб-интерфейс**.

Репозиторий: [github.com/nottmv/credit_scoring](https://github.com/nottmv/credit_scoring)

---

## Структура проекта

```
credit_scoring/
├── src/
│   ├── api/
│   │   ├── main.py                  # FastAPI: scoring, UI, retrain, drift
│   │   └── templates/
│   │       ├── dashboard.html       # Мониторинг + таблица предсказаний
│   │       ├── inference.html       # Страница инференса
│   │       └── experiments.html    # MLflow эксперименты
│   ├── models/
│   │   ├── train_model.py           # CatBoost / XGBoost, MLflow logging
│   │   └── mlflow_pyfunc.py         # MLflow pyfunc wrapper
│   ├── monitoring/
│   │   ├── drift.py                 # Data / target / concept drift (KS, PSI, z-test)
│   │   ├── metrics_store.py         # JSONL-лог событий scoring + feedback
│   │   └── prometheus_metrics.py    # Prometheus Counter, Gauge, Histogram
│   ├── features/
│   │   └── build_features.py        # Feature engineering
│   └── data/
│       └── make_dataset.py          # Загрузка сырых данных
├── scripts/
│   ├── drift_check.py               # CLI: data/target/concept drift report
│   ├── retrain_pipeline.py          # CLI: train + drift + MLflow
│   ├── download_drive_artifacts.py  # Скачать данные и модели с Google Drive
│   ├── start_local_stack.sh         # API + MLflow без Docker
│   ├── final_check.sh               # lint + tests + docker compose smoke
│   └── k8s_minikube.sh              # Деплой в minikube
├── k8s/
│   ├── deployment.yaml              # API Deployment
│   ├── service.yaml                 # API NodePort :30800
│   ├── mlflow.yaml                  # MLflow + PVC + Service :30500
│   ├── prometheus.yaml              # Prometheus ConfigMap + Deployment :30909
│   └── manual/
│       └── job-retrain.yaml         # Job переобучения (ручной apply)
├── argocd/
│   └── application.yaml            # ArgoCD Application (auto-sync k8s/)
├── monitoring/
│   ├── prometheus.yml               # Prometheus scrape config (docker compose)
│   └── grafana/
│       ├── provisioning/            # Datasource + dashboard provider
│       └── dashboards/
│           └── credit_scoring.json  # Grafana dashboard (9 панелей)
├── .github/workflows/
│   ├── ci.yml                       # lint → test → docker build → deploy
│   ├── commitlint.yml               # Проверка формата коммитов (Conventional Commits) при Pull Request
│   └── retrain-dispatch.yml         # Ручной запуск переобучения через Actions
├── docker/mlflow/Dockerfile         # MLflow tracking server image
├── Dockerfile                       # API image (python 3.11-slim)
├── docker-compose.yml               # API + MLflow + Prometheus + Grafana
├── data/
│   ├── raw/
│   │   ├── credit_scoring.csv.dvc   # Полный датасет (DVC → Google Drive)
│   │   └── synthetic_min.csv        # Минимальный синтетический (fallback)
│   └── ...
├── models/                          # .pkl (DVC tracked)
├── reports/                         # last_drift.json, champion_metrics.json, events.jsonl
├── tests/
│   ├── conftest.py                  # Fixtures: mock bundle, TestClient
│   ├── test_api.py                  # API tests (18 тестов)
│   └── test_drift.py                # Drift computation tests
├── Makefile                         # Основные команды
├── requirements.txt                 # Dev зависимости
├── requirements-docker.txt          # Runtime (без dev)
├── setup.py / setup.cfg             # Пакет src + flake8/pytest config
└── CONTRIBUTING.md                  # Git flow, conventional commits, Docker, K8s
```

---

## Датасет и модель

| Параметр | Значение |
|----------|---------|
| Датасет | [Give Me Some Credit (Kaggle)](https://www.kaggle.com/c/GiveMeSomeCredit) — 150k заёмщиков, 10 признаков |
| Цель | `Delinquent90` — просрочка 90+ дней в 2 года |
| Базовая модель | **CatBoostClassifier** (глубина 1, регуляризация, early stopping) |
| Альтернатива | **XGBoostBooster** (max_depth=1, alpha=50, lambda=200) |
| Метрика | ROC-AUC (Gini = 2·AUC − 1), разбивка train/test/validation |

---

## Быстрый старт

### 1. Клонирование и зависимости

```bash
git clone https://github.com/nottmv/credit_scoring.git
cd credit_scoring
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

### 2. Данные

**Вариант A — DVC:**
```bash
pip install 'dvc[gdrive]'
dvc pull  # OAuth Google при первом запуске
```

**Вариант B — прямое скачивание:**
```bash
python scripts/download_drive_artifacts.py
```

**Вариант C — обучение на синтетических данных (без авторизации):**
```bash
# make train автоматически использует synthetic_min.csv если нет credit_scoring.csv
make train
```

### 3. Обучение

```bash
make train            # CatBoost на доступных данных
make train-mlflow     # То же + логирование в MLflow (требует запущенный сервер)
```

### 4. Локальный запуск без Docker

```bash
make local-up
# API: http://127.0.0.1:8000
# MLflow: http://127.0.0.1:5000
```

### 5. Docker Compose (полный стек)

```bash
make train            # Нужна модель
make docker-up        # API + MLflow + Prometheus + Grafana
```

| Сервис | URL |
|--------|-----|
| API (Swagger) | http://localhost:8000/docs |
| Web UI (мониторинг) | http://localhost:8000/dashboard |
| Инференс | http://localhost:8000/ui/inference |
| Эксперименты | http://localhost:8000/ui/experiments |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

---

## Web UI

| Страница | Маршрут | Описание |
|----------|---------|---------|
| Dashboard | `/dashboard` | Метрики, таблица предсказаний с флагами аномалий, карточки дрейфа, кнопка переобучения |
| Инференс | `/ui/inference` | Форма ввода признаков, оценка вероятности дефолта, фидбек |
| Эксперименты | `/ui/experiments` | Запуски MLflow, champion-метрики, quick reference |

Аномалии: предсказания с вероятностью ≥ 0.8 помечаются как `Аномалия`.
Уведомления о дрейфе: баннер в шапке dashboard при `degraded = true` в drift-отчёте.

---

## API

| Метод | Маршрут | Описание |
|-------|---------|---------|
| GET | `/health` | Healthcheck |
| POST | `/v1/score` | Скоринг заёмщика |
| POST | `/v1/feedback` | Фидбек (реальный исход) |
| GET | `/v1/metrics/summary` | Агрегированные метрики (AUC, latency, drift) |
| GET | `/v1/predictions?limit=50` | Последние предсказания с флагами аномалий |
| GET | `/v1/drift/report` | Текущий drift-отчёт (JSON) |
| POST | `/internal/reload-model` | Hot-reload модели (X-Admin-Token) |
| POST | `/internal/retrain` | Запуск переобучения в фоне (X-Admin-Token) |
| GET | `/internal/retrain/status` | Статус фонового переобучения |
| GET | `/metrics` | Prometheus scrape endpoint |

---

## MLflow: графики и артефакты

При `make train-mlflow` (с Docker: `MLFLOW_URI=http://127.0.0.1:5001`) в каждый run логируются PNG в **Artifacts → plots/** (эксперимент `credit_scoring_served`):

| Файл | Содержание |
|------|------------|
| `roc_curves.png` | ROC-кривые train / test / validation |
| `pr_curves.png` | Precision–Recall |
| `score_distributions.png` | Распределение скоров по классам |
| `feature_importance.png` | Top-20 признаков |
| `learning_curve.png` | AUC по итерациям (CatBoost) |
| `metrics_comparison.png` | Сравнение ROC-AUC и Gini |

MLflow server запускается с `--serve-artifacts`, чтобы загрузка артефактов с хоста в Docker работала без ошибки `PermissionError: /mlflow`.

После изменения `docker/mlflow/Dockerfile` пересоберите образ: `make docker-build-mlflow && make docker-up`.

---

## Мониторинг дрейфа

```bash
# Генерация отчёта drift (data + target + concept)
make drift-check CURRENT=data/incoming/batch.csv

# Или напрямую:
python scripts/drift_check.py \
  --reference data/raw/credit_scoring.csv \
  --current data/incoming/batch.csv \
  --model-path models/model_bundle_catboost.pkl \
  --fail-on-drift
```

Отчёт сохраняется в `reports/last_drift.json` и отображается в `/dashboard` + Prometheus `/metrics`.

**Три типа дрейфа:**
- **Data drift** — KS-тест и PSI для каждого признака
- **Target drift** — z-тест для разницы долей дефолта (ref vs current)
- **Concept drift** — MAD матриц корреляции + KS распределений скоров

---

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`):
1. **lint** — `flake8` (max-line-length=100)
2. **test** — `pytest -q tests`
3. **docker build** — сборка образов API + MLflow (без push)
4. **deploy** — `kubectl apply` при push в `main` (если задан `KUBE_CONFIG_B64`)

Ручной запуск переобучения: Actions → **Retrain (manual)** → выбрать `model_type`.

---

## Kubernetes (Minikube)

```bash
# Один скрипт (нужен Docker + minikube + kubectl):
make minikube-deploy

# Проверка:
kubectl get pods
minikube service credit-scoring-api --url

# Job переобучения (после подготовки данных):
kubectl apply -f k8s/manual/job-retrain.yaml
```

NodePorts: API **30800**, MLflow **30500**, Prometheus **30909**.

---

## ArgoCD (CD в Kubernetes)

```bash
# Установка ArgoCD (если ещё нет):
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Деплой приложения:
kubectl apply -f argocd/application.yaml

# Веб-интерфейс:
kubectl port-forward svc/argocd-server -n argocd 8080:443
# Логин: admin, пароль: kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
```

ArgoCD будет автоматически синхронизировать состояние кластера с папкой `k8s/` репозитория (`automated.selfHeal: true`).

---

## Цикл деградации → переобучение → reload

```bash
# 1. Получили новые данные
python scripts/drift_check.py --current data/incoming/new_batch.csv --fail-on-drift

# 2. Если drift обнаружен — переобучаем
make retrain
# или через API:
curl -X POST http://localhost:8000/internal/retrain -H "X-Admin-Token: $ADMIN_RELOAD_TOKEN"

# 3. Hot-reload модели в сервисе
curl -X POST http://localhost:8000/internal/reload-model -H "X-Admin-Token: $ADMIN_RELOAD_TOKEN"

# 4. Мониторинг — drift отчёт обновлён, Grafana показывает метрики
```

---

## Тесты и линтер

```bash
# Lint
flake8 --max-line-length=100 --extend-exclude=.dvc src tests scripts

# Tests
pytest -q tests          # 22+ тестов: API, drift computation

# Финальная проверка (lint + tests + docker compose config):
make final-check

# С запуском контейнеров и curl всех URL:
make final-check-docker
```

---

## Шаблон Cookiecutter

Проект инициализирован на базе шаблона [**Cookiecutter Data Science**](https://drivendata.github.io/cookiecutter-data-science/):

```bash
cookiecutter gh:drivendata/cookiecutter-data-science
# project_name: credit_scoring, python_version: 3.11
```

Структура каталогов (`src/`, `data/`, `models/`, `reports/`, `Makefile`, `setup.py`, `docs/`) сохранена и расширена MLOps-компонентами (API, мониторинг, k8s, CI/CD).

---

## Git flow и конвенции

- **`main`** — стабильная ветка, деплой только через PR
- **`feature/<task>`**, **`fix/<task>`** — рабочие ветки
- [Conventional Commits](https://www.conventionalcommits.org/): `feat(api): batch score`, `fix(drift): PSI bins`, `ci: add retrain workflow`
- DVC для версионирования данных и моделей (`dvc pull` / `dvc push`)

---

## Авторы

Михаил Попов, Лия Суфиянова — ИТМО MLOps курс
