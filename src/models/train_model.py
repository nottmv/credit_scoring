"""Training and MLflow logging for credit scoring models."""

from __future__ import annotations

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

from src.models.shared import (
    CLIENT_ID_COL,
    TARGET,
    build_features,
)

try:
    import scorecardpy as sc
except Exception:
    sc = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


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


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape}")
    return df


def basic_eda(df: pd.DataFrame) -> None:
    print("\nDataset info:")
    print(df.info())

    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False))

    print("\nTarget distribution:")
    print(df[TARGET].value_counts(dropna=False))
    print(df[TARGET].value_counts(normalize=True, dropna=False))


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


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series, dataset_name: str = "dataset"):
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

    n_train = len(X_train)
    if n_train < 5000:
        depth = 4
        min_data_in_leaf = max(15, n_train // 20)
        l2_leaf_reg = 10.0
    else:
        depth = 1
        min_data_in_leaf = 500
        l2_leaf_reg = 120.0

    model = CatBoostClassifier(
        iterations=5000,
        depth=depth,
        learning_rate=0.015,
        l2_leaf_reg=l2_leaf_reg,
        random_strength=10.0,
        bagging_temperature=2.0,
        rsm=0.5,
        subsample=0.6,
        min_data_in_leaf=min_data_in_leaf,
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


def train_logreg_woe(df_train: pd.DataFrame):
    if sc is None:
        raise RuntimeError("scorecardpy is not available; cannot train WOE model")
    bins = sc.woebin(df_train, y=TARGET)
    df_train_woe = sc.woebin_ply(df_train, bins)

    X_train_woe = df_train_woe.drop(columns=[TARGET])
    y_train_woe = df_train_woe[TARGET]

    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train_woe, y_train_woe)

    return model, bins


def transform_woe(df: pd.DataFrame, bins: Any):
    df_woe = sc.woebin_ply(df, bins)
    X_woe = df_woe.drop(columns=[TARGET], errors="ignore")
    return X_woe


def save_model(model: Any, path: Union[str, Path]) -> None:
    joblib.dump(model, path)
    print(f"Model saved to {path}")


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


def train_and_evaluate(
    df_raw: pd.DataFrame,
    cfg: TrainConfig,
    model_type: str,
) -> Tuple[ModelBundle, Dict[str, Any], Dict[str, Tuple[pd.DataFrame, pd.Series]]]:
    X_train_raw, X_test_raw, X_val_raw, y_train, y_test, y_val = split_data_3way(
        df_raw, cfg
    )

    X_train, fit_params = build_features(X_train_raw, fit_params=None)
    X_test, _ = build_features(X_test_raw, fit_params=fit_params)
    X_val, _ = build_features(X_val_raw, fit_params=fit_params)

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


def main() -> None:
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
    model_types = ["catboost", "xgboost"] if args.model_type == "both" else [args.model_type]

    for mt in model_types:
        print(f"\n=== Training: {mt} ===")
        bundle, report, eval_sets = train_and_evaluate(df_raw, cfg, mt)

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


if __name__ == "__main__":
    main()
