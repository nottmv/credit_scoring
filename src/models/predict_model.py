"""Inference helpers for loading bundles and scoring raw model inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from src.models.shared import dataframe_from_input, prepare_features
from src.models.train_model import ModelBundle


def load_bundle(path: Union[str, Path]) -> ModelBundle:
    """Load a serialized model bundle from disk."""
    return ModelBundle.load(path)


def predict_proba(
    bundle: ModelBundle,
    model_input: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
) -> np.ndarray:
    """Return positive-class probabilities for raw input records."""
    df = dataframe_from_input(model_input)
    X = prepare_features(df, bundle)
    return bundle.model.predict_proba(X)[:, 1]


def predict_one(
    bundle: ModelBundle,
    model_input: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Return a single-row prediction payload."""
    proba = float(predict_proba(bundle, model_input)[0])
    return {
        "probability": proba,
        "model_type": bundle.model_type,
        "anomaly": proba >= 0.8,
    }
