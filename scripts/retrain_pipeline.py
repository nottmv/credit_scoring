#!/usr/bin/env python3
"""
Ручное / CI переобучение: train_model + опционально drift-check и MLflow.

После успешного обучения обновите модель в сервисе:
  curl -X POST http://localhost:8000/internal/reload-model -H "X-Admin-Token: $ADMIN_RELOAD_TOKEN"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
    print("+", " ".join(cmd))
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.check_call(cmd, cwd=str(ROOT), env=merged)


def main() -> int:
    p = argparse.ArgumentParser(description="Train + optional drift + MLflow")
    p.add_argument("--data", default="data/raw/credit_scoring.csv")
    p.add_argument("--model-type", default="catboost", choices=["catboost", "xgboost", "both"])
    p.add_argument("--save", default="models/model_bundle_catboost.pkl")
    p.add_argument("--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    p.add_argument("--mlflow-experiment", default="credit_scoring")
    p.add_argument("--mlflow-register", default=None)
    p.add_argument(
        "--drift-current",
        default=None,
        help="If set, run drift_check.py after training against this CSV",
    )
    p.add_argument("--fail-on-drift", action="store_true")
    args = p.parse_args()

    py = sys.executable
    train_cmd = [
        py,
        str(ROOT / "src/models/train_model.py"),
        "--data",
        args.data,
        "--model-type",
        args.model_type,
        "--save",
        args.save,
    ]
    env = os.environ.copy()
    if args.mlflow_uri:
        train_cmd += [
            "--mlflow-uri",
            args.mlflow_uri,
            "--mlflow-experiment",
            args.mlflow_experiment,
        ]
        if args.mlflow_register:
            train_cmd += ["--mlflow-register", args.mlflow_register]

    run(train_cmd, env=env)

    if args.drift_current:
        drift_cmd = [
            py,
            str(ROOT / "scripts/drift_check.py"),
            "--reference",
            args.data,
            "--current",
            args.drift_current,
            "--model-path",
            args.save,
            "--report-path",
            str(ROOT / "reports/last_drift.json"),
        ]
        if args.fail_on_drift:
            drift_cmd.append("--fail-on-drift")
        run(drift_cmd)

    print("\nReload API model (если сервис запущен):")
    print(
        '  curl -sS -X POST http://127.0.0.1:8000/internal/reload-model '
        '-H "X-Admin-Token: $ADMIN_RELOAD_TOKEN"'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
