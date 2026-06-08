import pandas as pd

from src.models.train_model import normalize_raw_columns


def test_age_alias_mapped_to_Age():
    df = pd.DataFrame([{"age": 40, "DebtRatio": 0.3}])
    out = normalize_raw_columns(df)
    assert "Age" in out.columns
    assert out["Age"].iloc[0] == 40
    assert "age" not in out.columns
