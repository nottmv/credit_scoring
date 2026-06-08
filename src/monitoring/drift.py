"""Data drift, target drift, and concept drift (distribution + structure)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, norm

from src.models.train_model import (
    CLIENT_ID_COL,
    TARGET,
    ModelBundle,
    build_features_for_training,
)

# --- PSI (population stability) ---


def _psi_single(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """PSI for one numeric column; expected = reference distribution sample."""
    e = expected[np.isfinite(expected)]
    a = actual[np.isfinite(actual)]
    if len(e) < 30 or len(a) < 10:
        return 0.0
    cuts = np.unique(np.quantile(e, np.linspace(0, 1, buckets + 1)))
    if len(cuts) < 3:
        return 0.0
    e_counts, _ = np.histogram(e, bins=cuts)
    a_counts, _ = np.histogram(a, bins=cuts)
    e_pct = np.clip(e_counts / max(e_counts.sum(), 1), 1e-6, 1.0)
    a_pct = np.clip(a_counts / max(a_counts.sum(), 1), 1e-6, 1.0)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


# --- Data drift (KS + PSI on numeric inputs) ---


@dataclass
class ColumnKS:
    column: str
    ks_statistic: float
    p_value: float
    alert: bool


@dataclass
class ColumnPSI:
    column: str
    psi: float
    alert: bool


@dataclass
class DataDriftResult:
    columns_evaluated: int
    max_ks: float
    max_psi: float
    ks_alerts: List[ColumnKS]
    psi_alerts: List[ColumnPSI]
    degraded: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "columns_evaluated": self.columns_evaluated,
            "max_ks": self.max_ks,
            "max_psi": self.max_psi,
            "ks_alerts": [asdict(x) for x in self.ks_alerts],
            "psi_alerts": [asdict(x) for x in self.psi_alerts],
            "degraded": self.degraded,
        }


def _numeric_shared_columns(
    ref: pd.DataFrame, cur: pd.DataFrame, exclude: Sequence[str]
) -> List[str]:
    ex = set(exclude)
    ref_num = ref.select_dtypes(include=[np.number]).columns
    cur_num = cur.select_dtypes(include=[np.number]).columns
    return [c for c in ref_num if c in cur_num and c not in ex]


def compute_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_col: str = TARGET,
    ks_alert_threshold: float = 0.2,
    psi_alert_threshold: float = 0.25,
    max_columns: int = 40,
) -> DataDriftResult:
    exclude = {target_col, CLIENT_ID_COL}
    cols = _numeric_shared_columns(reference, current, exclude)[:max_columns]
    ks_alerts: List[ColumnKS] = []
    psi_alerts: List[ColumnPSI] = []
    max_ks = 0.0
    max_psi = 0.0
    for col in cols:
        a = pd.to_numeric(reference[col], errors="coerce").dropna().to_numpy()
        b = pd.to_numeric(current[col], errors="coerce").dropna().to_numpy()
        if len(a) < 50 or len(b) < 50:
            continue
        stat, p = ks_2samp(a, b, alternative="two-sided", mode="auto")
        stat_f = float(stat)
        max_ks = max(max_ks, stat_f)
        if stat_f >= ks_alert_threshold:
            ks_alerts.append(
                ColumnKS(
                    column=col,
                    ks_statistic=round(stat_f, 6),
                    p_value=float(p),
                    alert=True,
                )
            )
        psi_v = _psi_single(a, b)
        max_psi = max(max_psi, psi_v)
        if psi_v >= psi_alert_threshold:
            psi_alerts.append(ColumnPSI(column=col, psi=round(psi_v, 6), alert=True))
    degraded = bool(ks_alerts or psi_alerts)
    return DataDriftResult(
        columns_evaluated=len(cols),
        max_ks=round(max_ks, 6),
        max_psi=round(max_psi, 6),
        ks_alerts=ks_alerts,
        psi_alerts=psi_alerts,
        degraded=degraded,
    )


# --- Target drift (prevalence of positive class) ---


@dataclass
class TargetDriftResult:
    reference_positive_rate: float
    current_positive_rate: float
    rate_diff: float
    z_score: Optional[float]
    p_value_two_sided: Optional[float]
    alert: bool
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def two_proportion_ztest(
    n1: int, x1: int, n2: int, x2: int
) -> Tuple[float, float]:
    """Z and two-sided p-value for difference in proportions."""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    if p_pool <= 0 or p_pool >= 1:
        return 0.0, 1.0
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p = float(2 * (1 - norm.cdf(abs(z))))
    return float(z), p


def compute_target_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_col: str = TARGET,
    z_alert: float = 3.0,
) -> TargetDriftResult:
    if target_col not in reference.columns or target_col not in current.columns:
        return TargetDriftResult(
            reference_positive_rate=0.0,
            current_positive_rate=0.0,
            rate_diff=0.0,
            z_score=None,
            p_value_two_sided=None,
            alert=False,
            note="Target column missing in reference or current; skipped.",
        )
    y1 = pd.to_numeric(reference[target_col], errors="coerce").dropna()
    y2 = pd.to_numeric(current[target_col], errors="coerce").dropna()
    y1 = y1.astype(int)
    y2 = y2.astype(int)
    n1, n2 = len(y1), len(y2)
    x1, x2 = int(y1.sum()), int(y2.sum())
    r1 = x1 / n1 if n1 else 0.0
    r2 = x2 / n2 if n2 else 0.0
    z, p = two_proportion_ztest(n1, x1, n2, x2)
    alert = abs(z) >= z_alert if n1 > 20 and n2 > 20 else False
    return TargetDriftResult(
        reference_positive_rate=round(r1, 6),
        current_positive_rate=round(r2, 6),
        rate_diff=round(r2 - r1, 6),
        z_score=round(z, 4) if n1 and n2 else None,
        p_value_two_sided=round(p, 6) if p is not None else None,
        alert=alert,
        note="Two-sample z-test on default rate; alert if |z| >= %.1f" % z_alert,
    )


# --- Concept drift ---


@dataclass
class ConceptDriftResult:
    correlation_mean_abs_diff: Optional[float]
    score_distribution_ks: Optional[float]
    score_mean_ref: Optional[float]
    score_mean_cur: Optional[float]
    alert: bool
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _corr_mad(ref: pd.DataFrame, cur: pd.DataFrame, cols: List[str]) -> float:
    if len(cols) < 2:
        return 0.0
    R1 = ref[cols].corr().values
    R2 = cur[cols].corr().values
    diff = np.nanmean(np.abs(R1 - R2))
    return float(diff)


def _scores_for_frame(bundle: ModelBundle, df_raw: pd.DataFrame) -> np.ndarray:
    df = df_raw.copy()
    target = bundle.config.get("target_col", TARGET)
    if target in df.columns:
        df_fit = df.drop(columns=[target])
    else:
        df_fit = df
    X, _ = build_features_for_training(df_fit, fit_params=bundle.preprocessing_params)
    X = X.reindex(columns=bundle.feature_cols, fill_value=0)
    return bundle.model.predict_proba(X)[:, 1]


def compute_concept_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    bundle: Optional[ModelBundle] = None,
    target_col: str = TARGET,
    corr_alert_threshold: float = 0.12,
    score_ks_alert_threshold: float = 0.15,
    max_corr_cols: int = 25,
) -> ConceptDriftResult:
    exclude = {target_col, CLIENT_ID_COL}
    cols = _numeric_shared_columns(reference, current, exclude)[:max_corr_cols]
    corr_diff = _corr_mad(reference, current, cols) if len(cols) >= 2 else None

    score_ks = None
    mean_ref = mean_cur = None
    if bundle is not None:
        try:
            s_ref = _scores_for_frame(bundle, reference)
            s_cur = _scores_for_frame(bundle, current)
            if len(s_ref) > 50 and len(s_cur) > 50:
                score_ks, _ = ks_2samp(s_ref, s_cur, alternative="two-sided", mode="auto")
                score_ks = float(score_ks)
                mean_ref = float(np.mean(s_ref))
                mean_cur = float(np.mean(s_cur))
        except Exception as e:
            return ConceptDriftResult(
                correlation_mean_abs_diff=round(corr_diff, 6) if corr_diff else None,
                score_distribution_ks=None,
                score_mean_ref=None,
                score_mean_cur=None,
                alert=bool(corr_diff and corr_diff >= corr_alert_threshold),
                note=f"Score-based concept drift skipped: {e}",
            )

    alert_corr = corr_diff is not None and corr_diff >= corr_alert_threshold
    alert_score = score_ks is not None and score_ks >= score_ks_alert_threshold
    return ConceptDriftResult(
        correlation_mean_abs_diff=round(corr_diff, 6) if corr_diff is not None else None,
        score_distribution_ks=round(score_ks, 6) if score_ks is not None else None,
        score_mean_ref=round(mean_ref, 6) if mean_ref is not None else None,
        score_mean_cur=round(mean_cur, 6) if mean_cur is not None else None,
        alert=bool(alert_corr or alert_score),
        note=(
            "Concept: correlation structure vs ref (MAD of corr matrices); "
            "if model given — KS of score distributions (prior shift / leakage proxy)."
        ),
    )


@dataclass
class FullDriftReport:
    reference_rows: int
    current_rows: int
    data_drift: DataDriftResult
    target_drift: TargetDriftResult
    concept_drift: ConceptDriftResult

    @property
    def degraded(self) -> bool:
        return (
            self.data_drift.degraded
            or self.target_drift.alert
            or self.concept_drift.alert
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference_rows": self.reference_rows,
            "current_rows": self.current_rows,
            "degraded": self.degraded,
            "data_drift": self.data_drift.to_dict(),
            "target_drift": self.target_drift.to_dict(),
            "concept_drift": self.concept_drift.to_dict(),
        }


def compute_full_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    bundle: Optional[ModelBundle] = None,
    target_col: str = TARGET,
    ks_alert_threshold: float = 0.2,
    psi_alert_threshold: float = 0.25,
) -> FullDriftReport:
    return FullDriftReport(
        reference_rows=len(reference),
        current_rows=len(current),
        data_drift=compute_data_drift(
            reference,
            current,
            target_col=target_col,
            ks_alert_threshold=ks_alert_threshold,
            psi_alert_threshold=psi_alert_threshold,
        ),
        target_drift=compute_target_drift(reference, current, target_col=target_col),
        concept_drift=compute_concept_drift(
            reference, current, bundle=bundle, target_col=target_col
        ),
    )


def save_drift_report(report: FullDriftReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def load_drift_report(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# Backwards-compatible name used in tests / older scripts
def compute_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    ks_alert_threshold: float = 0.2,
    max_columns: int = 40,
) -> "LegacyDriftReport":
    d = compute_data_drift(
        reference,
        current,
        ks_alert_threshold=ks_alert_threshold,
        psi_alert_threshold=999.0,
        max_columns=max_columns,
    )
    return LegacyDriftReport(
        reference_rows=len(reference),
        current_rows=len(current),
        columns_evaluated=d.columns_evaluated,
        alerts=d.ks_alerts,
        max_ks=d.max_ks,
        degraded=d.degraded,
    )


@dataclass
class LegacyDriftReport:
    reference_rows: int
    current_rows: int
    columns_evaluated: int
    alerts: List[ColumnKS]
    max_ks: float
    degraded: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference_rows": self.reference_rows,
            "current_rows": self.current_rows,
            "columns_evaluated": self.columns_evaluated,
            "max_ks": self.max_ks,
            "degraded": self.degraded,
            "alerts": [asdict(a) for a in self.alerts],
        }
