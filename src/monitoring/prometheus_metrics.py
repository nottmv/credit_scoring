"""Prometheus instruments; drift gauges refreshed from JSON on /metrics scrape."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from prometheus_client import Counter, Gauge, Histogram, generate_latest

SCORE_REQUESTS = Counter(
    "credit_score_scoring_requests_total",
    "Total scoring API calls",
)
SCORE_ERRORS = Counter(
    "credit_score_scoring_errors_total",
    "Scoring failures (validation / model)",
)
SCORE_LATENCY = Histogram(
    "credit_score_scoring_latency_seconds",
    "Scoring latency",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PREDICTION_PROBA = Histogram(
    "credit_score_prediction_probability",
    "Predicted probability of default",
    buckets=[i / 20 for i in range(21)],
)

GAUGE_DATA_MAX_KS = Gauge(
    "credit_drift_data_max_ks",
    "Max KS statistic across features (from last drift report)",
)
GAUGE_DATA_MAX_PSI = Gauge(
    "credit_drift_data_max_psi",
    "Max PSI across features",
)
GAUGE_TARGET_RATE_REF = Gauge(
    "credit_drift_target_positive_rate_reference",
    "Reference positive rate",
)
GAUGE_TARGET_RATE_CUR = Gauge(
    "credit_drift_target_positive_rate_current",
    "Current positive rate",
)
GAUGE_TARGET_Z = Gauge(
    "credit_drift_target_z_score",
    "Z-score for target prevalence shift",
)
GAUGE_CONCEPT_CORR_MAD = Gauge(
    "credit_drift_concept_correlation_mean_abs_diff",
    "Mean abs diff of correlation matrices",
)
GAUGE_CONCEPT_SCORE_KS = Gauge(
    "credit_drift_concept_score_distribution_ks",
    "KS between model score distributions ref vs current",
)
GAUGE_DRIFT_DEGRADED = Gauge(
    "credit_drift_degraded",
    "1 if any drift dimension flagged",
)


def observe_score_ok(latency_seconds: float, probability: float) -> None:
    SCORE_REQUESTS.inc()
    SCORE_LATENCY.observe(latency_seconds)
    PREDICTION_PROBA.observe(probability)


def observe_score_error() -> None:
    SCORE_ERRORS.inc()


def _nested_get(d: Dict[str, Any], *keys: str, default: Optional[float] = None) -> Optional[float]:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    if isinstance(cur, (int, float)) and cur == cur:
        return float(cur)
    return default


def refresh_drift_gauges(drift_path: Path) -> None:
    if not drift_path.is_file():
        return
    try:
        d = json.loads(drift_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return

    # New nested format
    if "data_drift" in d:
        GAUGE_DATA_MAX_KS.set(_nested_get(d, "data_drift", "max_ks") or 0.0)
        GAUGE_DATA_MAX_PSI.set(_nested_get(d, "data_drift", "max_psi") or 0.0)
        z = _nested_get(d, "target_drift", "z_score")
        GAUGE_TARGET_Z.set(z if z is not None else 0.0)
        GAUGE_TARGET_RATE_REF.set(
            _nested_get(d, "target_drift", "reference_positive_rate") or 0.0
        )
        GAUGE_TARGET_RATE_CUR.set(
            _nested_get(d, "target_drift", "current_positive_rate") or 0.0
        )
        cm = _nested_get(d, "concept_drift", "correlation_mean_abs_diff")
        GAUGE_CONCEPT_CORR_MAD.set(cm if cm is not None else 0.0)
        sk = _nested_get(d, "concept_drift", "score_distribution_ks")
        GAUGE_CONCEPT_SCORE_KS.set(sk if sk is not None else 0.0)
        GAUGE_DRIFT_DEGRADED.set(1.0 if d.get("degraded") else 0.0)
        return

    # Legacy flat report (only KS)
    GAUGE_DATA_MAX_KS.set(float(d.get("max_ks", 0.0)))
    GAUGE_DATA_MAX_PSI.set(0.0)
    GAUGE_TARGET_Z.set(0.0)
    GAUGE_TARGET_RATE_REF.set(0.0)
    GAUGE_TARGET_RATE_CUR.set(0.0)
    GAUGE_CONCEPT_CORR_MAD.set(0.0)
    GAUGE_CONCEPT_SCORE_KS.set(0.0)
    GAUGE_DRIFT_DEGRADED.set(1.0 if d.get("degraded") else 0.0)


def metrics_payload(drift_path: Path) -> bytes:
    refresh_drift_gauges(drift_path)
    return generate_latest()
