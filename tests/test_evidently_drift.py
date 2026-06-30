"""Tests for Evidently drift report generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.monitoring.evidently_drift import generate_evidently_report


def test_generate_evidently_report_creates_html(tmp_path: Path) -> None:
    ref = pd.read_csv("data/raw/synthetic_min.csv").head(400)
    cur = ref.copy()
    cur["Age"] = cur["Age"] + 10
    html = tmp_path / "report.html"
    summary = generate_evidently_report(ref, cur, html_path=html)
    assert html.is_file()
    assert html.stat().st_size > 1000
    assert "feature_psi" in summary
    assert summary["reference_rows"] == 400
