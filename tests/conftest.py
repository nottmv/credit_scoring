"""Shared fixtures for the test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ensure project root is in PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("ADMIN_RELOAD_TOKEN", "test-token")
os.environ.setdefault("MODEL_PATH", str(ROOT / "models" / "model_bundle_catboost.pkl"))


def _make_mock_bundle() -> MagicMock:
    """Create a minimal ModelBundle mock that returns fixed predictions."""
    bundle = MagicMock()
    bundle.model_type = "catboost"
    bundle.feature_cols = ["RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio"]
    bundle.preprocessing_params = {"medians": {}, "quantiles": {}}
    bundle.config = {"target_col": "Delinquent90"}

    mock_model = MagicMock()
    mock_model.predict_proba = MagicMock(
        return_value=np.array([[0.75, 0.25]])
    )
    bundle.model = mock_model
    return bundle


@pytest.fixture()
def mock_bundle() -> MagicMock:
    return _make_mock_bundle()


@pytest.fixture()
def api_client(tmp_path: Path, mock_bundle: MagicMock) -> Generator:
    """TestClient with mocked model bundle and temporary report paths."""
    events_path = tmp_path / "events.jsonl"
    drift_path = tmp_path / "last_drift.json"
    champion_path = tmp_path / "champion_metrics.json"

    with (
        patch("src.api.main.load_bundle", return_value=mock_bundle),
        patch("src.api.main._bundle", mock_bundle),
        patch("src.api.main.EVENTS_PATH", events_path),
        patch("src.api.main.DRIFT_PATH", drift_path),
        patch("src.api.main.CHAMPION_PATH", champion_path),
        patch(
            "src.models.train_model.build_features_for_training",
            return_value=(
                pd.DataFrame(
                    [[0.3, 45, 0.35]],
                    columns=["RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio"],
                ),
                {},
            ),
        ),
    ):
        from src.api.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


@pytest.fixture()
def sample_features() -> dict:
    return {
        "RevolvingUtilizationOfUnsecuredLines": 0.3,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.35,
        "MonthlyIncome": 6000,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 0,
    }
