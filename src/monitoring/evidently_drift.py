"""Evidently AI reports: data drift, target drift, concept drift (HTML + JSON summary)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.models.shared import CLIENT_ID_COL, TARGET

# Numeric features used in Evidently reports (Kaggle Give Me Some Credit schema)
FEATURE_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "Age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


def _prepare(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = reference.copy()
    cur = current.copy()
    if CLIENT_ID_COL in ref.columns:
        ref = ref.drop(columns=[CLIENT_ID_COL])
    if CLIENT_ID_COL in cur.columns:
        cur = cur.drop(columns=[CLIENT_ID_COL])
    cols = [c for c in FEATURE_COLUMNS if c in ref.columns and c in cur.columns]
    if TARGET in ref.columns and TARGET in cur.columns:
        cols = cols + [TARGET]
    return ref[cols], cur[cols]


def generate_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    html_path: Path,
    json_path: Optional[Path] = None,
    prediction_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build Evidently Report (data + target + concept presets) and save HTML/JSON.
    Returns a compact summary dict for Prometheus / API.
    """
    from evidently import Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    ref, cur = _prepare(reference, current)
    presets = [DataSummaryPreset(), DataDriftPreset()]
    report = Report(presets)
    snapshot = report.run(reference_data=ref, current_data=cur)

    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))

    summary = _extract_summary(snapshot, ref, cur)
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _extract_summary(snapshot: Any, ref: pd.DataFrame, cur: pd.DataFrame) -> Dict[str, Any]:
    """Map Evidently snapshot metrics to our monitoring schema."""
    feature_psi: Dict[str, float] = {}
    max_psi = 0.0
    drifted_features: List[str] = []

    try:
        for test in snapshot.dict().get("tests", []):
            name = str(test.get("name", ""))
            if "Drift" in name and test.get("status") == "FAIL":
                params = test.get("parameters", {}) or {}
                col = params.get("column") or params.get("feature")
                if col:
                    drifted_features.append(str(col))
    except Exception:
        pass

    # PSI from column-level statistics when available
    try:
        metrics = snapshot.dict().get("metrics", [])
        for m in metrics:
            mid = str(m.get("metric_id", ""))
            if "DriftValue" in mid or "PSI" in mid.upper():
                params = m.get("value", {}) or m.get("params", {})
                if isinstance(params, dict):
                    for col, val in params.items():
                        if isinstance(val, (int, float)):
                            feature_psi[str(col)] = float(val)
                            max_psi = max(max_psi, float(val))
    except Exception:
        pass

    # Fallback: compute PSI via existing module for top features
    if not feature_psi:
        from src.monitoring.drift import compute_data_drift

        dd = compute_data_drift(ref, cur)
        for alert in dd.psi_alerts:
            feature_psi[alert.column] = alert.psi
            max_psi = max(max_psi, alert.psi)
        for alert in dd.ks_alerts:
            if alert.column not in drifted_features:
                drifted_features.append(alert.column)

    ref_rate = float(ref[TARGET].mean()) if TARGET in ref.columns else 0.0
    cur_rate = float(cur[TARGET].mean()) if TARGET in cur.columns else 0.0

    from src.monitoring.drift import compute_full_drift_report

    scipy_report = compute_full_drift_report(ref, cur)
    degraded = scipy_report.degraded or bool(drifted_features) or max_psi >= 0.25

    return {
        "source": "evidently",
        "reference_rows": len(ref),
        "current_rows": len(cur),
        "degraded": degraded,
        "max_psi": round(max_psi, 6),
        "drifted_features": drifted_features,
        "feature_psi": {k: round(v, 6) for k, v in feature_psi.items()},
        "target_drift": {
            "reference_positive_rate": round(ref_rate, 6),
            "current_positive_rate": round(cur_rate, 6),
            "rate_diff": round(cur_rate - ref_rate, 6),
        },
        "data_drift": scipy_report.data_drift.to_dict(),
        "concept_drift": scipy_report.concept_drift.to_dict(),
        "scipy_target_drift": scipy_report.target_drift.to_dict(),
    }
