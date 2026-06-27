"""MLflow pyfunc wrapper for ModelRegistry from joblib ModelBundle."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import mlflow.pyfunc
import numpy as np
import pandas as pd

from src.models.predict_model import predict_proba
from src.models.train_model import ModelBundle


class CreditBundleModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.bundle = ModelBundle.load(context.artifacts["bundle"])

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
    ) -> np.ndarray:
        return predict_proba(self.bundle, model_input)


def log_pyfunc_model(
    bundle_path: str,
    registered_model_name: Optional[str] = None,
    artifact_path: str = "credit_model",
    input_example: Optional[pd.DataFrame] = None,
) -> Any:
    kwargs: Dict[str, Any] = {
        "python_model": CreditBundleModel(),
        "artifact_path": artifact_path,
        "artifacts": {"bundle": bundle_path},
        "registered_model_name": registered_model_name,
    }
    if input_example is not None:
        kwargs["input_example"] = input_example
    return mlflow.pyfunc.log_model(**kwargs)
