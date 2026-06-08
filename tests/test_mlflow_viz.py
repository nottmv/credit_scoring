"""Tests for training plot generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.mlflow_viz import generate_training_plots
from src.models.train_model import TrainConfig, train_and_evaluate


def test_generate_training_plots_creates_pngs(tmp_path: Path):
    data = Path("data/raw/synthetic_min.csv")
    if not data.is_file():
        return

    df = pd.read_csv(data)
    bundle, report, eval_sets = train_and_evaluate(df, TrainConfig(), "catboost")
    out = tmp_path / "plots"
    paths = generate_training_plots(bundle, report, eval_sets, out)

    assert len(paths) >= 4
    for p in paths:
        assert p.is_file()
        assert p.suffix == ".png"
        assert p.stat().st_size > 500

    names = {p.name for p in paths}
    assert "roc_curves.png" in names
    assert "metrics_comparison.png" in names
