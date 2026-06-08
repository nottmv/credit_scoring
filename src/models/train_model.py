import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import scorecardpy as sc
except Exception:
    sc = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


TARGET = "Delinquent90"
CLIENT_ID_COL = "client_id"
RANDOM_STATE = 42
TIME_COL_CANDIDATES = ["snapshot_date", "report_date", "as_of_date", "date", "dt"]


@dataclass(frozen=True)
class TrainConfig:
    target_col: str = TARGET
    id_col: str = CLIENT_ID_COL
    random_state: int = RANDOM_STATE
    validation_size: float = 0.15
    test_size: float = 0.15
    time_col: Optional[str] = None


@dataclass(frozen=True)
class ModelBundle:
    model_type: str
    model: Any
    feature_cols: List[str]
    preprocessing_params: Dict[str, Any]
    config: Dict[str, Any]

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": self.model_type,
            "model_payload": _serialize_model(self.model_type, self.model),
            "feature_cols": self.feature_cols,
            "preprocessing_params": self.preprocessing_params,
            "config": self.config,
        }
        joblib.dump(payload, path)

    @staticmethod
    def load(path: Union[str, Path]) -> "ModelBundle":
        payload = joblib.load(path)
        model_type = payload["model_type"]
        model = _deserialize_model(model_type, payload["model_payload"])
        return ModelBundle(
            model_type=model_type,
            model=model,
            feature_cols=list(payload["feature_cols"]),
            preprocessing_params=dict(payload.get("preprocessing_params", {})),
            config=dict(payload["config"]),
        )


def _serialize_model(model_type: str, model: Any) -> Any:
    if model_type == "xgboost":
        import xgboost as xgb  # type: ignore

        booster = getattr(model, "booster", None)
        if booster is None and isinstance(model, xgb.Booster):
            booster = model
        if booster is None:
            raise ValueError("xgboost model is expected to have a Booster")
        return {"kind": "booster_raw", "raw": booster.save_raw()}
    return {"kind": "joblib", "obj": model}


def _deserialize_model(model_type: str, payload: Any) -> Any:
    kind = payload.get("kind")
    if model_type == "xgboost":
        if kind != "booster_raw":
            raise ValueError("Unexpected xgboost payload kind")
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(payload["raw"])
        return XGBBoosterWrapper(booster)
    if kind == "joblib":
        return payload["obj"]
    raise ValueError("Unknown model payload kind")


def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape}")
    return df


def basic_eda(df):
    print("\nDataset info:")
    print(df.info())

    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False))

    print("\nTarget distribution:")
    print(df[TARGET].value_counts(dropna=False))
    print(df[TARGET].value_counts(normalize=True, dropna=False))


def preprocess_data(df, fit_params=None):
    df = df.copy()

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET in numeric_cols:
        numeric_cols.remove(TARGET)

    if fit_params is None:
        # Режим обучения: вычисляем медианы и границы клиппинга
        medians = {}
        quantiles = {}
        for col in numeric_cols:
            # Замена inf
            col_clean = df[col].replace([np.inf, -np.inf], np.nan)
            q01 = col_clean.quantile(0.01)
            q99 = col_clean.quantile(0.99)
            if pd.notna(q01) and pd.notna(q99):
                quantiles[col] = (q01, q99)
            else:
                quantiles[col] = (np.nan, np.nan)
            med = col_clean.median()
            medians[col] = med if pd.notna(med) else 0.0
        fit_params = {'medians': medians, 'quantiles': quantiles}
    else:
        medians = fit_params['medians']
        quantiles = fit_params['quantiles']

    # Применяем преобразования
    for col in numeric_cols:
        # Замена inf
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        # Клиппинг
        q01, q99 = quantiles.get(col, (np.nan, np.nan))
        if pd.notna(q01) and pd.notna(q99):
            df[col] = df[col].clip(q01, q99)
        # Заполнение пропусков
        median_val = medians.get(col, 0.0)
        df[col] = df[col].fillna(median_val)

    # Объектные колонки – без изменений (заполняем "missing")
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in object_cols:
        df[col] = df[col].fillna("missing")

    return df, fit_params


def add_features(df):
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


def select_model_columns(df):
    df = df.copy()

    drop_cols = []

    if CLIENT_ID_COL in df.columns:
        drop_cols.append(CLIENT_ID_COL)

    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    drop_cols.extend(object_cols)

    drop_cols = list(set(drop_cols))
    usable_cols = [col for col in df.columns if col not in drop_cols]

    return df[usable_cols]


def _detect_time_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in cols_lower:
            return cols_lower[c]
    return None


def split_data_3way(
    df: pd.DataFrame,
    cfg: TrainConfig,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column {cfg.target_col} not found")

    time_col = cfg.time_col or _detect_time_col(df, TIME_COL_CANDIDATES)

    if time_col and time_col in df.columns:
        df2 = df.copy()
        df2[time_col] = pd.to_datetime(df2[time_col], errors="coerce")
        df2 = df2.sort_values(time_col)
        n_val = int(np.ceil(len(df2) * cfg.validation_size))
        if n_val < 1:
            raise ValueError("Validation split too small")
        df_val = df2.iloc[-n_val:].copy()
        df_rest = df2.iloc[:-n_val].copy()

        X_val = df_val.drop(columns=[cfg.target_col])
        y_val = df_val[cfg.target_col]

        X_rest = df_rest.drop(columns=[cfg.target_col])
        y_rest = df_rest[cfg.target_col]

        test_frac_of_rest = cfg.test_size / (1.0 - cfg.validation_size)
        X_train, X_test, y_train, y_test = train_test_split(
            X_rest,
            y_rest,
            test_size=test_frac_of_rest,
            stratify=y_rest,
            random_state=cfg.random_state,
        )
    else:
        X = df.drop(columns=[cfg.target_col])
        y = df[cfg.target_col]

        test_plus_val = cfg.test_size + cfg.validation_size
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=test_plus_val,
            stratify=y,
            random_state=cfg.random_state,
        )
        val_frac_of_temp = cfg.validation_size / test_plus_val
        X_test, X_val, y_test, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_frac_of_temp,
            stratify=y_temp,
            random_state=cfg.random_state,
        )

    print("\nSplit sizes:")
    print("Train:", X_train.shape, y_train.shape)
    print("Test:", X_test.shape, y_test.shape)
    print("Validation (prod-like):", X_val.shape, y_val.shape)

    return X_train, X_test, X_val, y_train, y_test, y_val


def evaluate_model(model, X, y, dataset_name="dataset"):
    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    gini = 2 * auc - 1

    result = {
        "dataset": dataset_name,
        "roc_auc": round(float(auc), 6),
        "gini": round(float(gini), 6),
    }

    print(result)
    return result


class XGBBoosterWrapper:
    def __init__(self, booster: Any):
        self.booster = booster

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb  # type: ignore

        d = xgb.DMatrix(X)
        p = self.booster.predict(d)
        p = np.clip(p, 0.0, 1.0)
        return np.vstack([1.0 - p, p]).T


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame, y_eval: pd.Series
):

    import xgboost as xgb

    dtrain = xgb.DMatrix(X_train, label=y_train)
    deval = xgb.DMatrix(X_eval, label=y_eval)

    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    params = {
        "objective": "binary:logistic",
        "max_depth": 1,
        "eta": 0.01,
        "subsample": 0.5,
        "colsample_bytree": 0.5,
        "min_child_weight": 150,
        "gamma": 2.0,
        "alpha": 50.0,
        "lambda": 200.0,
        "max_delta_step": 1,
        "eval_metric": "auc",
        "scale_pos_weight": scale_pos_weight,
        "seed": RANDOM_STATE,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=[(deval, "eval")],
        early_stopping_rounds=200,
        verbose_eval=False,
    )
    return XGBBoosterWrapper(booster)


def train_catboost(
    X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame, y_eval: pd.Series
):

    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    class_weights = [1.0, (neg / pos) if pos > 0 else 1.0]

    model = CatBoostClassifier(
        iterations=5000,
        depth=1,
        learning_rate=0.015,
        l2_leaf_reg=120,
        random_strength=10.0,
        bagging_temperature=2.0,
        rsm=0.5,
        subsample=0.6,
        min_data_in_leaf=500,
        class_weights=class_weights,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=0,
        od_type="Iter",
        od_wait=200,
    )

    model.fit(X_train, y_train, eval_set=(X_eval, y_eval), use_best_model=True)
    return model


def train_logreg_woe(df_train):
    if sc is None:
        raise RuntimeError("scorecardpy is not available; cannot train WOE model")
    bins = sc.woebin(df_train, y=TARGET)
    df_train_woe = sc.woebin_ply(df_train, bins)

    X_train_woe = df_train_woe.drop(columns=[TARGET])
    y_train_woe = df_train_woe[TARGET]

    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train_woe, y_train_woe)

    return model, bins


def transform_woe(df, bins):
    df_woe = sc.woebin_ply(df, bins)
    X_woe = df_woe.drop(columns=[TARGET], errors="ignore")
    return X_woe


def save_model(model, path):
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def predict_by_client_id(bundle: ModelBundle, df_raw, client_id):
    if CLIENT_ID_COL not in df_raw.columns:
        raise ValueError(f"Column {CLIENT_ID_COL} not found in dataset")

    row = df_raw[df_raw[CLIENT_ID_COL] == client_id].copy()
    if row.empty:
        raise ValueError("Client not found")

    row, _ = build_features_for_training(row, fit_params=bundle.preprocessing_params)
    if bundle.config.get("target_col") in row.columns:
        row = row.drop(columns=[bundle.config["target_col"]])
    row = row.reindex(columns=bundle.feature_cols, fill_value=0)

    proba = bundle.model.predict_proba(row)[:, 1][0]

    return {"client_id": client_id, "probability": float(proba)}


def _overfit_report(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_name = {m["dataset"]: m for m in metrics}
    if not {"train", "test", "validation"} <= set(by_name.keys()):
        return {"note": "insufficient datasets for overfit report"}
    return {
        "train_test_auc_gap": round(
            by_name["train"]["roc_auc"] - by_name["test"]["roc_auc"], 6
        ),
        "test_val_auc_gap": round(
            by_name["test"]["roc_auc"] - by_name["validation"]["roc_auc"], 6
        ),
        "train_test_gini_gap": round(
            by_name["train"]["gini"] - by_name["test"]["gini"], 6
        ),
        "test_val_gini_gap": round(
            by_name["test"]["gini"] - by_name["validation"]["gini"], 6
        ),
    }


def build_features_for_training(df_raw, fit_params=None):
    df, new_params = preprocess_data(df_raw, fit_params)
    df = add_features(df)
    df = select_model_columns(df)
    return df, new_params


def train_and_evaluate(
    df_raw: pd.DataFrame,
    cfg: TrainConfig,
    model_type: str,
) -> Tuple[ModelBundle, Dict[str, Any], Dict[str, Tuple[pd.DataFrame, pd.Series]]]:
    X_train_raw, X_test_raw, X_val_raw, y_train, y_test, y_val = (
        split_data_3way(df_raw, cfg)
    )

    X_train, fit_params = build_features_for_training(X_train_raw, fit_params=None)
    X_test, _ = build_features_for_training(X_test_raw, fit_params=fit_params)
    X_val, _ = build_features_for_training(X_val_raw, fit_params=fit_params)

    feature_cols = X_train.columns.tolist()
    X_test = X_test[feature_cols]
    X_val = X_val[feature_cols]

    if model_type == "xgboost":
        model = train_xgboost(X_train, y_train, X_test, y_test)
    elif model_type == "catboost":
        model = train_catboost(X_train, y_train, X_test, y_test)
    else:
        raise ValueError("model_type must be one of: xgboost, catboost")

    metrics = [
        evaluate_model(model, X_train, y_train, "train"),
        evaluate_model(model, X_test, y_test, "test"),
        evaluate_model(model, X_val, y_val, "validation"),
    ]
    report = {
        "model_type": model_type,
        "metrics": metrics,
        "overfit": _overfit_report(metrics),
    }
    print("\nOverfit check:")
    print(report["overfit"])

    eval_sets = {
        "train": (X_train, y_train),
        "test": (X_test, y_test),
        "validation": (X_val, y_val),
    }

    bundle = ModelBundle(
        model_type=model_type,
        model=model,
        feature_cols=feature_cols,
        preprocessing_params=fit_params,
        config=asdict(cfg),
    )
    return bundle, report, eval_sets


def score_client_id(
    bundle: ModelBundle, df_raw: pd.DataFrame, client_id: Any
) -> Dict[str, Any]:

    model = bundle.model

    if CLIENT_ID_COL not in df_raw.columns:
        raise ValueError(f"Column {CLIENT_ID_COL} not found in dataset")

    row = df_raw[df_raw[CLIENT_ID_COL] == client_id].copy()
    if row.empty:
        raise ValueError("Client not found")

    row, _ = build_features_for_training(row, fit_params=bundle.preprocessing_params)
    if bundle.config.get("target_col") in row.columns:
        row = row.drop(columns=[bundle.config["target_col"]])
    row = row.reindex(columns=bundle.feature_cols, fill_value=0)
    proba = float(model.predict_proba(row)[:, 1][0])

    try:
        client_id_out = client_id.item()
    except Exception:
        client_id_out = client_id
    return {"client_id": client_id_out, "probability": proba}


def write_champion_metrics(report: Dict[str, Any], path: Union[str, Path]) -> None:
    """Snapshot of offline validation metrics (merged per model_type for monitoring)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    val = next((m for m in report["metrics"] if m["dataset"] == "validation"), None)
    test = next((m for m in report["metrics"] if m["dataset"] == "test"), None)
    mt = report["model_type"]
    payload = {
        "model_type": mt,
        "validation_roc_auc": val["roc_auc"] if val else None,
        "test_roc_auc": test["roc_auc"] if test else None,
        "overfit": report.get("overfit", {}),
    }
    store: Dict[str, Any] = {"models": {}}
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "models" in raw:
                store = raw
        except json.JSONDecodeError:
            pass
    store.setdefault("models", {})[mt] = payload
    path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_mlflow_experiment(tracking_uri: str, experiment_name: str) -> str:
    """Use an experiment whose artifacts upload via the tracking server (mlflow-artifacts:/)."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    exp = client.get_experiment_by_name(experiment_name)
    if exp is not None and str(exp.artifact_location).startswith("mlflow-artifacts:"):
        return exp.experiment_id

    served_name = f"{experiment_name}_served"
    served = client.get_experiment_by_name(served_name)
    if served is None:
        return client.create_experiment(
            served_name,
            artifact_location="mlflow-artifacts:/",
        )
    return served.experiment_id


def log_mlflow_run(
    report: Dict[str, Any],
    bundle: ModelBundle,
    bundle_path: Path,
    eval_sets: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    tracking_uri: str,
    experiment_name: str,
    register_name: Optional[str],
) -> None:
    import mlflow

    from src.models.mlflow_pyfunc import log_pyfunc_model
    from src.models.mlflow_viz import log_training_plots

    experiment_id = _ensure_mlflow_experiment(tracking_uri, experiment_name)
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param("model_type", report["model_type"])
        for m in report["metrics"]:
            prefix = m["dataset"]
            mlflow.log_metric(f"{prefix}_roc_auc", m["roc_auc"])
            mlflow.log_metric(f"{prefix}_gini", m["gini"])
        for key, val in report.get("overfit", {}).items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                mlflow.log_metric(key, float(val))

        plot_artifacts = log_training_plots(bundle, report, eval_sets)
        mlflow.log_param("plot_artifacts_count", len(plot_artifacts))

        test_X, _ = eval_sets["test"]
        input_example = test_X.head(1)
        log_pyfunc_model(
            str(bundle_path.resolve()),
            registered_model_name=register_name,
            input_example=input_example,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/credit_scoring.csv")
    parser.add_argument(
        "--model-type",
        choices=["xgboost", "catboost", "both"],
        default="both",
    )
    parser.add_argument(
        "--time-col",
        default=None,
        help="Optional time column for prod-like validation split",
    )
    parser.add_argument("--save", default="models/model_bundle.pkl")
    parser.add_argument(
        "--score-client-id",
        default=None,
        help="If set, outputs probability for this id",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI (logs metrics + pyfunc when set)",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="credit_scoring",
    )
    parser.add_argument(
        "--mlflow-register",
        default=None,
        help="Optional Model Registry name",
    )
    parser.add_argument(
        "--champion-report",
        default="reports/champion_metrics.json",
        help="Write offline validation baseline for monitoring UI",
    )
    args = parser.parse_args()

    print(f"\nModel type: {args.model_type}")
    cfg = TrainConfig(time_col=args.time_col)
    df_raw = load_data(args.data)
    basic_eda(df_raw)

    save_path = Path(args.save)
    model_types = (
        ["catboost", "xgboost"] if args.model_type == "both" else [args.model_type]
    )

    bundles: Dict[str, ModelBundle] = {}
    for mt in model_types:
        print(f"\n=== Training: {mt} ===")
        bundle, report, eval_sets = train_and_evaluate(df_raw, cfg, mt)
        bundles[mt] = bundle

        out_path = save_path
        if args.model_type == "both":
            if save_path.suffix.lower() == ".pkl":
                out_path = save_path.with_name(f"{save_path.stem}_{mt}.pkl")
            else:
                out_path = save_path / f"model_bundle_{mt}.pkl"

        bundle.save(out_path)
        print(f"\nSaved bundle to {out_path}")
        print("\nTrain report (json):")
        print(json.dumps(report, ensure_ascii=False, indent=2))

        write_champion_metrics(report, args.champion_report)
        print(f"\nChampion / baseline metrics written to {args.champion_report}")

        if args.mlflow_uri:
            log_mlflow_run(
                report,
                bundle,
                out_path,
                eval_sets,
                args.mlflow_uri,
                args.mlflow_experiment,
                args.mlflow_register,
            )
            print(f"\nMLflow: logged run to {args.mlflow_uri}")

    if args.score_client_id is not None:
        client_id = type(df_raw[CLIENT_ID_COL].iloc[0])(args.score_client_id)
        for mt, bundle in bundles.items():
            result = score_client_id(bundle, df_raw, client_id)
            print(f"\nScore result ({mt}):")
            print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
