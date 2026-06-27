"""Shared preprocessing and feature engineering utilities for training and inference."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

TARGET = "Delinquent90"
CLIENT_ID_COL = "client_id"


def normalize_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common API aliases to training CSV column names."""
    df = df.copy()
    aliases = {
        "age": "Age",
    }
    for src, dst in aliases.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
            df = df.drop(columns=[src])
    return df


def preprocess_data(
    df: pd.DataFrame,
    fit_params: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET in numeric_cols:
        numeric_cols.remove(TARGET)

    if fit_params is None:
        medians: Dict[str, float] = {}
        quantiles: Dict[str, Tuple[float, float]] = {}
        for col in numeric_cols:
            col_clean = df[col].replace([np.inf, -np.inf], np.nan)
            q01 = col_clean.quantile(0.01)
            q99 = col_clean.quantile(0.99)
            if pd.notna(q01) and pd.notna(q99):
                quantiles[col] = (float(q01), float(q99))
            else:
                quantiles[col] = (np.nan, np.nan)
            med = col_clean.median()
            medians[col] = float(med) if pd.notna(med) else 0.0
        fit_params = {"medians": medians, "quantiles": quantiles}
    else:
        medians = fit_params["medians"]
        quantiles = fit_params["quantiles"]

    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        q01, q99 = quantiles.get(col, (np.nan, np.nan))
        if pd.notna(q01) and pd.notna(q99):
            df[col] = df[col].clip(q01, q99)
        median_val = medians.get(col, 0.0)
        df[col] = df[col].fillna(median_val)

    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in object_cols:
        df[col] = df[col].fillna("missing")

    return df, fit_params


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if (
        "NumberOfTime30-59DaysPastDueNotWorse" in df.columns
        and "NumberOfTime60-89DaysPastDueNotWorse" in df.columns
        and "NumberOfTimes90DaysLate" in df.columns
    ):
        df["total_past_due"] = (
            df["NumberOfTime30-59DaysPastDueNotWorse"]
            + df["NumberOfTime60-89DaysPastDueNotWorse"]
            + df["NumberOfTimes90DaysLate"]
        )
        df["any_past_due_flag"] = (df["total_past_due"] > 0).astype(int)
        df["severe_past_due_flag"] = (df["NumberOfTimes90DaysLate"] > 0).astype(int)

    if "RevolvingUtilizationOfUnsecuredLines" in df.columns:
        df["high_util_flag"] = (
            df["RevolvingUtilizationOfUnsecuredLines"] > 0.8
        ).astype(int)
        df["very_high_util_flag"] = (
            df["RevolvingUtilizationOfUnsecuredLines"] > 1.0
        ).astype(int)
        df["log_revolving_util"] = np.log1p(
            np.clip(df["RevolvingUtilizationOfUnsecuredLines"], a_min=0, a_max=None)
        )

    if "DebtRatio" in df.columns and "MonthlyIncome" in df.columns:
        df["debt_income_ratio"] = df["DebtRatio"] * df["MonthlyIncome"]
        df["utilization_times_debt"] = df["DebtRatio"]
        if "RevolvingUtilizationOfUnsecuredLines" in df.columns:
            df["utilization_times_debt"] = (
                df["RevolvingUtilizationOfUnsecuredLines"] * df["DebtRatio"]
            )
        df["log_income"] = np.log1p(np.clip(df["MonthlyIncome"], a_min=0, a_max=None))
        df["log_debt_ratio"] = np.log1p(np.clip(df["DebtRatio"], a_min=0, a_max=None))
        df["income_is_missing_flag"] = (df["MonthlyIncome"] <= 0).astype(int)

    if "MonthlyIncome" in df.columns and "NumberOfDependents" in df.columns:
        df["income_per_person"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
        df["has_dependents_flag"] = (df["NumberOfDependents"] > 0).astype(int)

    if "NumberOfOpenCreditLinesAndLoans" in df.columns and "DebtRatio" in df.columns:
        df["debt_per_loan"] = df["DebtRatio"] / (
            df["NumberOfOpenCreditLinesAndLoans"] + 1
        )
        df["many_trade_lines_flag"] = (
            df["NumberOfOpenCreditLinesAndLoans"] >= 10
        ).astype(int)

    if (
        "NumberRealEstateLoansOrLines" in df.columns
        and "NumberOfOpenCreditLinesAndLoans" in df.columns
    ):
        df["real_estate_share"] = df["NumberRealEstateLoansOrLines"] / (
            df["NumberOfOpenCreditLinesAndLoans"] + 1
        )
        df["has_real_estate_flag"] = (df["NumberRealEstateLoansOrLines"] > 0).astype(
            int
        )

    if "Age" in df.columns:
        df["age_squared"] = df["Age"] ** 2
        df["young_borrower_flag"] = (df["Age"] < 30).astype(int)
        df["senior_borrower_flag"] = (df["Age"] >= 60).astype(int)

    return df


def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    drop_cols: List[str] = []

    if CLIENT_ID_COL in df.columns:
        drop_cols.append(CLIENT_ID_COL)

    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    drop_cols.extend(object_cols)

    drop_cols = list(set(drop_cols))
    usable_cols = [col for col in df.columns if col not in drop_cols]
    return df[usable_cols]


def build_features(
    df_raw: pd.DataFrame,
    fit_params: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df_raw = normalize_raw_columns(df_raw)
    df, new_params = preprocess_data(df_raw, fit_params)
    df = add_features(df)
    df = select_model_columns(df)
    return df, new_params


def dataframe_from_input(
    model_input: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]
) -> pd.DataFrame:
    if isinstance(model_input, dict):
        return pd.DataFrame([model_input])
    if isinstance(model_input, list):
        return pd.DataFrame(model_input)
    return model_input.copy()


def prepare_features(df_raw: pd.DataFrame, bundle: Any) -> pd.DataFrame:
    target = bundle.config.get("target_col")
    df = df_raw.copy()
    if target and target in df.columns:
        df = df.drop(columns=[target])
    df_feat, _ = build_features(df, fit_params=bundle.preprocessing_params)
    return df_feat.reindex(columns=bundle.feature_cols, fill_value=0)
