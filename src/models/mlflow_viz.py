"""Training evaluation plots logged as MLflow artifacts."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from src.models.train_model import ModelBundle

EvalSets = Dict[str, Tuple[pd.DataFrame, pd.Series]]


def _predict_proba(bundle: ModelBundle, X: pd.DataFrame) -> np.ndarray:
    return bundle.model.predict_proba(X)[:, 1]


def plot_roc_curves(
    bundle: ModelBundle,
    eval_sets: EvalSets,
    out_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (X, y) in eval_sets.items():
        proba = _predict_proba(bundle, X)
        fpr, tpr, _ = roc_curve(y, proba)
        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{name} (AUC={auc(fpr, tpr):.4f})",
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.35, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    path = out_dir / "roc_curves.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pr_curves(
    bundle: ModelBundle,
    eval_sets: EvalSets,
    out_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (X, y) in eval_sets.items():
        proba = _predict_proba(bundle, X)
        precision, recall, _ = precision_recall_curve(y, proba)
        ax.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{name} (AP={auc(recall, precision):.4f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curves")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    path = out_dir / "pr_curves.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_score_distribution(
    bundle: ModelBundle,
    eval_sets: EvalSets,
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, len(eval_sets), figsize=(5 * len(eval_sets), 4))
    if len(eval_sets) == 1:
        axes = [axes]
    for ax, (name, (X, y)) in zip(axes, eval_sets.items()):
        proba = _predict_proba(bundle, X)
        y_arr = np.asarray(y)
        ax.hist(
            proba[y_arr == 0],
            bins=30,
            alpha=0.6,
            label="class 0",
            density=True,
            color="#2a9d8f",
        )
        ax.hist(
            proba[y_arr == 1],
            bins=30,
            alpha=0.6,
            label="class 1",
            density=True,
            color="#e63946",
        )
        ax.set_title(f"Score distribution — {name}")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = out_dir / "score_distributions.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(bundle: ModelBundle, out_dir: Path) -> Path | None:
    importances: np.ndarray
    if bundle.model_type == "catboost":
        importances = np.asarray(bundle.model.get_feature_importance(), dtype=float)
        labels = list(bundle.feature_cols)
    elif bundle.model_type == "xgboost":
        booster = bundle.model.booster
        score_map = booster.get_score(importance_type="gain")
        labels = list(bundle.feature_cols)
        importances = np.array(
            [float(score_map.get(f, score_map.get(f"f{i}", 0.0))) for i, f in enumerate(labels)],
            dtype=float,
        )
    else:
        return None

    if importances.sum() <= 0:
        return None

    top_n = min(20, len(labels))
    order = np.argsort(importances)[::-1][:top_n]
    top_labels = [labels[i] for i in order]
    top_vals = importances[order]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * top_n)))
    y_pos = np.arange(top_n)
    ax.barh(y_pos, top_vals, color="#1d3557", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} features — {bundle.model_type}")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "feature_importance.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metrics_bar(report: Dict[str, Any], out_dir: Path) -> Path:
    metrics = report.get("metrics", [])
    names = [m["dataset"] for m in metrics]
    aucs = [m["roc_auc"] for m in metrics]
    ginis = [m["gini"] for m in metrics]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, aucs, width, label="ROC-AUC", color="#457b9d")
    ax.bar(x + width / 2, ginis, width, label="Gini", color="#e9c46a")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Offline metrics — {report.get('model_type', 'model')}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(aucs):
        ax.text(i - width / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    path = out_dir / "metrics_comparison.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_learning_curve(bundle: ModelBundle, out_dir: Path) -> Path | None:
    evals: Dict[str, Dict[str, List[float]]] | None = None
    if bundle.model_type == "catboost" and hasattr(bundle.model, "get_evals_result"):
        evals = bundle.model.get_evals_result()
    elif bundle.model_type == "xgboost" and hasattr(bundle.model, "booster"):
        # XGBoost stores eval history on booster attributes when available
        booster = bundle.model.booster
        if hasattr(booster, "attr") and booster.attr("best_score") is not None:
            return None

    if not evals:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for dataset_name, metrics in evals.items():
        for metric_name, values in metrics.items():
            if "AUC" in metric_name.upper() or "auc" in metric_name:
                ax.plot(values, label=f"{dataset_name} — {metric_name}", linewidth=2)
                plotted = True
    if not plotted:
        plt.close(fig)
        return None

    ax.set_xlabel("Iteration")
    ax.set_ylabel("AUC")
    ax.set_title("Learning curve (eval AUC)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "learning_curve.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_training_plots(
    bundle: ModelBundle,
    report: Dict[str, Any],
    eval_sets: EvalSets,
    out_dir: Path,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = [
        plot_roc_curves(bundle, eval_sets, out_dir),
        plot_pr_curves(bundle, eval_sets, out_dir),
        plot_score_distribution(bundle, eval_sets, out_dir),
        plot_metrics_bar(report, out_dir),
    ]
    fi = plot_feature_importance(bundle, out_dir)
    if fi is not None:
        paths.append(fi)
    lc = plot_learning_curve(bundle, out_dir)
    if lc is not None:
        paths.append(lc)
    return paths


def log_training_plots(
    bundle: ModelBundle,
    report: Dict[str, Any],
    eval_sets: EvalSets,
) -> List[str]:
    """Log PNG plots to the active MLflow run under artifact path ``plots/``."""
    import mlflow

    with tempfile.TemporaryDirectory(prefix="mlflow_plots_") as tmp:
        tmp_path = Path(tmp)
        plot_paths = generate_training_plots(bundle, report, eval_sets, tmp_path)
        mlflow.log_artifacts(str(tmp_path), artifact_path="plots")
    return [f"plots/{p.name}" for p in plot_paths]
