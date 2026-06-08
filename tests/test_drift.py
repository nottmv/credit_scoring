import numpy as np
import pandas as pd

from src.monitoring.drift import (
    compute_concept_drift,
    compute_data_drift,
    compute_full_drift_report,
    compute_target_drift,
)


def _synth(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = (rng.random(n) < 0.12).astype(int)
    return pd.DataFrame({"f1": x, "f2": x * 0.5 + rng.normal(0, 0.1, n), "Delinquent90": y})


def test_data_drift_stable():
    ref = _synth(800, seed=1)
    cur = _synth(800, seed=2)
    r = compute_data_drift(ref, cur, ks_alert_threshold=0.5, psi_alert_threshold=1.0)
    assert r.columns_evaluated >= 1
    assert r.max_ks < 0.5


def test_target_drift_detects_shift():
    ref = pd.DataFrame({"Delinquent90": [0] * 900 + [1] * 100})
    cur = pd.DataFrame({"Delinquent90": [0] * 700 + [1] * 300})
    t = compute_target_drift(ref, cur, z_alert=2.0)
    assert t.alert is True


def test_concept_correlation_only():
    ref = _synth(500, seed=3)
    cur = ref.copy()
    cur["f2"] = -3.0 * ref["f1"]
    c = compute_concept_drift(ref, cur, bundle=None, corr_alert_threshold=0.01)
    assert c.correlation_mean_abs_diff is not None
    assert c.correlation_mean_abs_diff > 0.01


def test_full_report_dict():
    ref = _synth(600, seed=4)
    cur = _synth(600, seed=5)
    rep = compute_full_drift_report(ref, cur, bundle=None)
    d = rep.to_dict()
    assert "data_drift" in d and "target_drift" in d and "concept_drift" in d
