FROM python:3.11-slim

WORKDIR /mlflow

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir mlflow==3.1.4

ENV MLFLOW_BACKEND_STORE_URI=sqlite:////mlflow/mlflow.db \
    MLFLOW_ARTIFACTS_DESTINATION=/mlflow/artifacts \
    MLFLOW_DEFAULT_ARTIFACT_ROOT=mlflow-artifacts:/

EXPOSE 5000

CMD ["sh", "-c", "mkdir -p ${MLFLOW_ARTIFACTS_DESTINATION} && exec mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} --default-artifact-root ${MLFLOW_DEFAULT_ARTIFACT_ROOT} --artifacts-destination ${MLFLOW_ARTIFACTS_DESTINATION} --serve-artifacts"]
