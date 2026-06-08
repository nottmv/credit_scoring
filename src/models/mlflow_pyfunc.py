"""MLflow pyfunc wrapper for ModelRegistry from joblib ModelBundle."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import mlflow.pyfunc
import numpy as np
import pandas as pd

from src.models.train_model import ModelBundle, build_features_for_training


class CreditBundleModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.bundle = ModelBundle.load(context.artifacts["bundle"])

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
    ) -> np.ndarray:
        if isinstance(model_input, dict):
            df = pd.DataFrame([model_input])
        elif isinstance(model_input, list):
            df = pd.DataFrame(model_input)
        else:
            df = model_input.copy()

        target = self.bundle.config.get("target_col")
        if target and target in df.columns:
            df = df.drop(columns=[target])

        df_feat, _ = build_features_for_training(
            df, fit_params=self.bundle.preprocessing_params
        )
        df_feat = df_feat.reindex(columns=self.bundle.feature_cols, fill_value=0)
        return self.bundle.model.predict_proba(df_feat)[:, 1]


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
