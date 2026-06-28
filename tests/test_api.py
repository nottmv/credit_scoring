"""API integration tests (no real model required — uses mock bundle)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ── /health ────────────────────────────────────────────────────────────────────


def test_health_ok(api_client: TestClient) -> None:
    r = api_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_type" in body


# ── /v1/score ──────────────────────────────────────────────────────────────────


def test_score_returns_probability(api_client: TestClient, sample_features: dict) -> None:
    r = api_client.post("/v1/score", json={"features": sample_features})
    assert r.status_code == 200
    body = r.json()
    assert "probability" in body
    assert 0.0 <= body["probability"] <= 1.0
    assert "request_id" in body
    assert "model_type" in body
    assert "anomaly" in body


def test_score_anomaly_flag_low(api_client: TestClient, sample_features: dict) -> None:
    """Probability 0.25 should NOT trigger anomaly flag."""
    r = api_client.post("/v1/score", json={"features": sample_features})
    assert r.status_code == 200
    body = r.json()
    assert body["probability"] == pytest.approx(0.25, abs=0.01)
    assert body["anomaly"] is False


def test_score_anomaly_flag_high(tmp_path: Path, sample_features: dict) -> None:
    """Probability ≥ 0.8 should trigger anomaly flag."""
    from src.api.main import app

    bundle = MagicMock()
    bundle.model_type = "catboost"
    bundle.feature_cols = list(sample_features.keys())[:3]
    bundle.preprocessing_params = {"medians": {}, "quantiles": {}}
    bundle.config = {"target_col": "Delinquent90"}
    bundle.model.predict_proba = MagicMock(return_value=np.array([[0.1, 0.9]]))

    with (
        patch("src.api.main.load_bundle", return_value=bundle),
        patch("src.api.main._bundle", bundle),
        patch("src.api.main.EVENTS_PATH", tmp_path / "e.jsonl"),
        patch(
            "src.models.shared.build_features",
            return_value=(pd.DataFrame([[0.3, 45, 0.35]], columns=bundle.feature_cols), {}),
        ),
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.post("/v1/score", json={"features": sample_features})
    assert r.status_code == 200
    assert r.json()["anomaly"] is True


# ── /v1/feedback ───────────────────────────────────────────────────────────────


def test_feedback_accepted(api_client: TestClient, sample_features: dict) -> None:
    score_resp = api_client.post("/v1/score", json={"features": sample_features})
    rid = score_resp.json()["request_id"]
    r = api_client.post("/v1/feedback", json={"request_id": rid, "y_true": 0})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_feedback_invalid_label(api_client: TestClient) -> None:
    r = api_client.post("/v1/feedback", json={"request_id": "xyz", "y_true": 99})
    assert r.status_code == 422


# ── /v1/metrics/summary ────────────────────────────────────────────────────────


def test_metrics_summary_structure(api_client: TestClient) -> None:
    r = api_client.get("/v1/metrics/summary")
    assert r.status_code == 200
    body = r.json()
    for key in ("n_scores", "n_feedback", "production_roc_auc"):
        assert key in body


# ── /v1/predictions ────────────────────────────────────────────────────────────


def test_predictions_endpoint(api_client: TestClient, sample_features: dict) -> None:
    api_client.post("/v1/score", json={"features": sample_features})
    r = api_client.get("/v1/predictions")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


# ── /v1/drift/report ──────────────────────────────────────────────────────────


def test_drift_report_no_file(api_client: TestClient) -> None:
    r = api_client.get("/v1/drift/report")
    assert r.status_code == 200
    body = r.json()
    assert "error" in body or isinstance(body, dict)


def test_drift_report_with_file(api_client: TestClient, tmp_path: Path) -> None:
    drift_data = {
        "degraded": False,
        "reference_rows": 100,
        "current_rows": 100,
        "data_drift": {"max_ks": 0.05, "max_psi": 0.01, "degraded": False,
                       "ks_alerts": [], "psi_alerts": [], "columns_evaluated": 5},
        "target_drift": {"alert": False},
        "concept_drift": {"alert": False},
    }
    drift_file = tmp_path / "last_drift.json"
    drift_file.write_text(json.dumps(drift_data))
    with patch("src.api.main.DRIFT_PATH", drift_file):
        r = api_client.get("/v1/drift/report")
    assert r.status_code == 200
    assert r.json()["degraded"] is False


# ── /internal/reload-model ────────────────────────────────────────────────────


def test_reload_model_requires_token(api_client: TestClient) -> None:
    r = api_client.post("/internal/reload-model")
    assert r.status_code in (403, 503)


def test_reload_model_with_valid_token(api_client: TestClient) -> None:
    r = api_client.post(
        "/internal/reload-model",
        headers={"X-Admin-Token": "test-token"},
    )
    # May be 200 (model file found) or 400 (no file in test env) — not auth error
    assert r.status_code in (200, 400)


def test_reload_model_wrong_token(api_client: TestClient) -> None:
    r = api_client.post(
        "/internal/reload-model",
        headers={"X-Admin-Token": "wrong"},
    )
    assert r.status_code == 403


# ── /metrics (Prometheus) ─────────────────────────────────────────────────────


def test_prometheus_metrics_endpoint(api_client: TestClient) -> None:
    r = api_client.get("/metrics")
    assert r.status_code == 200
    assert "credit_score" in r.text or "HELP" in r.text


# ── Dashboard / UI ────────────────────────────────────────────────────────────


def test_dashboard_returns_html(api_client: TestClient) -> None:
    r = api_client.get("/dashboard")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_ui_inference_returns_html(api_client: TestClient) -> None:
    r = api_client.get("/ui/inference")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_ui_experiments_returns_html(api_client: TestClient) -> None:
    r = api_client.get("/ui/experiments")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


# ── OpenAPI ───────────────────────────────────────────────────────────────────


def test_openapi_schema(api_client: TestClient) -> None:
    r = api_client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert schema["info"]["title"].startswith("Credit scoring")
    paths = schema["paths"]
    assert "/v1/score" in paths
    assert "/v1/feedback" in paths
    assert "/health" in paths
