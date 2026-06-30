.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3 \
	test fetch-data train train-mlflow drift-check retrain api local-up \
	docker-build docker-build-mlflow docker-up docker-down docker-ps final-check final-check-docker \
	k8s-images minikube-deploy

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = credit_scoring
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 --max-line-length=100 --extend-exclude=.dvc src tests scripts

## Run unit tests
test: requirements
	$(PYTHON_INTERPRETER) -m pytest -q tests

## Download data + models from Google Drive (requires gdown)
fetch-data: requirements
	$(PYTHON_INTERPRETER) -m pip install -q gdown
	$(PYTHON_INTERPRETER) scripts/download_drive_artifacts.py

## Train CatBoost (credit_scoring.csv или synthetic_min.csv)
train: requirements
	@if [ -f data/raw/credit_scoring.csv ]; then \
		$(PYTHON_INTERPRETER) src/models/train_model.py --data data/raw/credit_scoring.csv --model-type catboost --save models/model_bundle_catboost.pkl; \
	else \
		echo ">>> data/raw/credit_scoring.csv нет — обучение на data/raw/synthetic_min.csv (make fetch-data для полного датасета)"; \
		$(PYTHON_INTERPRETER) src/models/train_model.py --data data/raw/synthetic_min.csv --model-type catboost --save models/model_bundle_catboost.pkl; \
	fi

## То же + лог в MLflow (сервер :5000, см. make local-up). MLFLOW_URI=http://host:5000 make train-mlflow
MLFLOW_URI ?= http://127.0.0.1:5000
train-mlflow:
	PYTHONPATH=. bash -c '\
		if [ -f data/raw/credit_scoring.csv ]; then DATA=data/raw/credit_scoring.csv; \
		else DATA=data/raw/synthetic_min.csv; fi && \
		$(PYTHON_INTERPRETER) src/models/train_model.py --data $$DATA --model-type catboost \
			--save models/model_bundle_catboost.pkl \
			--mlflow-uri $(MLFLOW_URI) --mlflow-experiment credit_scoring \
			--mlflow-register credit_scoring_catboost'

## Drift report: make drift-check CURRENT=data/incoming/batch.csv
drift-check: requirements
	@test -n "$(CURRENT)" || (echo "Usage: make drift-check CURRENT=path/to/current.csv"; exit 1)
	PYTHONPATH=. bash -c '\
		if [ -f data/raw/credit_scoring.csv ]; then REF=data/raw/credit_scoring.csv; \
		else REF=data/raw/synthetic_min.csv; fi && \
		$(PYTHON_INTERPRETER) scripts/drift_check.py --reference $$REF --current $(CURRENT) --model-path models/model_bundle_catboost.pkl'

## Drift simulator (docker compose profile or default stack)
drift-sim: requirements
	PYTHONPATH=. $(PYTHON_INTERPRETER) scripts/drift_simulator.py --interval 30

## DVC pipeline (train + drift_check)
dvc-repro: requirements
	dvc repro

## DVC push/pull to local remote (.dvc-storage)
dvc-push:
	dvc push -r localstorage

dvc-pull:
	dvc pull -r localstorage

## Retrain (optional MLflow: export MLFLOW_TRACKING_URI=...)
retrain: requirements
	@if [ -f data/raw/credit_scoring.csv ]; then \
		$(PYTHON_INTERPRETER) scripts/retrain_pipeline.py --data data/raw/credit_scoring.csv --model-type catboost --save models/model_bundle_catboost.pkl; \
	else \
		$(PYTHON_INTERPRETER) scripts/retrain_pipeline.py --data data/raw/synthetic_min.csv --model-type catboost --save models/model_bundle_catboost.pkl; \
	fi

## FastAPI locally
api: requirements
	cd $(PROJECT_DIR) && PYTHONPATH=. MODEL_PATH=$(PROJECT_DIR)/models/model_bundle_catboost.pkl $(PYTHON_INTERPRETER) -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

## API + MLflow locally (no Docker); logs in /tmp/credit_scoring_*.log
local-up: requirements
	chmod +x scripts/start_local_stack.sh
	test -f models/model_bundle_catboost.pkl || $(MAKE) train
	./scripts/start_local_stack.sh

## Docker images
docker-build:
	docker build -t credit-scoring-api:latest .

docker-build-mlflow:
	docker build -f docker/mlflow/Dockerfile -t credit-scoring-mlflow:latest .

k8s-images: docker-build docker-build-mlflow

## Docker Compose: нужен Docker Desktop / Colima + файл models/model_bundle_catboost.pkl
docker-up:
	@test -f models/model_bundle_catboost.pkl || (echo "Сначала: make train  или  make fetch-data && make train"; exit 1)
	docker compose up --build -d

docker-down:
	docker compose down

docker-ps:
	docker compose ps

## Финальная проверка (lint, тесты, docker compose config); без запуска контейнеров
final-check:
	chmod +x scripts/final_check.sh
	./scripts/final_check.sh

## То же + подъём docker compose и curl всех сервисов (порты подбираются, если 5000/8000 заняты)
final-check-docker:
	chmod +x scripts/final_check.sh
	./scripts/final_check.sh --docker

## Kubernetes (minikube): сборка образов в minikube docker + apply
minikube-deploy:
	chmod +x scripts/k8s_minikube.sh
	./scripts/k8s_minikube.sh

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
