#!/usr/bin/env python3
"""Regenerate data/raw/synthetic_min.csv with labels correlated to risk features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "synthetic_min.csv"
RNG = np.random.default_rng(42)
N = 800


def main() -> None:
    util = RNG.uniform(0.05, 1.2, N)
    age = RNG.integers(21, 68, N)
    past30 = RNG.integers(0, 5, N)
    past60 = RNG.integers(0, 3, N)
    past90 = RNG.integers(0, 3, N)
    debt = RNG.uniform(0.1, 1.8, N)
    income = RNG.exponential(4500, N) + 500
    lines = RNG.integers(1, 14, N)
    re_lines = RNG.integers(0, 4, N)
    deps = RNG.integers(0, 5, N)

    logit = (
        -6.2
        + 1.2 * util
        + 0.8 * past30
        + 1.0 * past60
        + 1.3 * past90
        + 0.5 * debt
        - 0.02 * age
        - 0.00005 * income
        + 0.1 * deps
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    target = (RNG.random(N) < prob).astype(int)

    df = pd.DataFrame(
        {
            "client_id": np.arange(N),
            "Delinquent90": target,
            "MonthlyIncome": income,
            "Age": age,
            "NumberOfOpenCreditLinesAndLoans": lines,
            "DebtRatio": debt,
            "RevolvingUtilizationOfUnsecuredLines": util,
            "NumberOfDependents": deps,
            "NumberRealEstateLoansOrLines": re_lines,
            "NumberOfTime30-59DaysPastDueNotWorse": past30,
            "NumberOfTime60-89DaysPastDueNotWorse": past60,
            "NumberOfTimes90DaysLate": past90,
        }
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(df)} rows, default rate {target.mean():.3f})")


if __name__ == "__main__":
    main()
