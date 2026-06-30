#!/usr/bin/env python3
"""
Synthetic drift stream: gradually shifts feature distributions and target rate,
runs drift checks on an interval, updates reports/ + Prometheus gauges via API.

Usage:
  python scripts/drift_simulator.py --interval 30
  DRIFT_SIM_INTERVAL=30 docker compose up drift-simulator
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.shared import TARGET  # noqa: E402
from src.monitoring.drift import compute_full_drift_report, save_drift_report  # noqa: E402
from src.monitoring.evidently_drift import generate_evidently_report  # noqa: E402


def load_reference(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Reference data not found: {path}")
    return pd.read_csv(path)


def make_drifted_batch(reference: pd.DataFrame, step: int, batch_size: int = 500) -> pd.DataFrame:
    """
    Simulate progressive drift:
    - steps 0-2: stable (subset of reference)
    - step 3+: shift Age (+5y/step), MonthlyIncome (-10%/step), target rate (+2%/step)
    """
    rng = np.random.default_rng(42 + step)
    idx = rng.choice(len(reference), size=min(batch_size, len(reference)), replace=True)
    batch = reference.iloc[idx].copy().reset_index(drop=True)

    if step >= 3:
        age_shift = min((step - 2) * 5, 25)
        batch["Age"] = (batch["Age"] + age_shift).clip(18, 100)
        income_factor = max(0.5, 1.0 - 0.1 * (step - 2))
        batch["MonthlyIncome"] = batch["MonthlyIncome"] * income_factor
        batch["RevolvingUtilizationOfUnsecuredLines"] = (
            batch["RevolvingUtilizationOfUnsecuredLines"] * (1.0 + 0.05 * (step - 2))
        ).clip(0, 2)
        flip_rate = min(0.02 * (step - 2), 0.15)
        flip_mask = rng.random(len(batch)) < flip_rate
        batch.loc[flip_mask, TARGET] = 1 - batch.loc[flip_mask, TARGET]

    return batch


def run_once(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    model_path: Path | None,
    reports_dir: Path,
) -> dict:
    bundle = None
    if model_path and model_path.is_file():
        from src.models.train_model import ModelBundle

        bundle = ModelBundle.load(model_path)

    report = compute_full_drift_report(reference, current, bundle=bundle)
    save_drift_report(report, reports_dir / "last_drift.json")

    evidently_summary = generate_evidently_report(
        reference,
        current,
        html_path=reports_dir / "drift_report.html",
        json_path=reports_dir / "evidently_drift.json",
    )
    evidently_summary["degraded"] = report.degraded or evidently_summary.get("degraded", False)
    (reports_dir / "evidently_drift.json").write_text(
        __import__("json").dumps(evidently_summary, indent=2),
        encoding="utf-8",
    )
    return report.to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic drift simulator loop")
    parser.add_argument(
        "--reference",
        default=os.environ.get("DRIFT_REFERENCE", "data/raw/synthetic_min.csv"),
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", "models/model_bundle_catboost.pkl"),
    )
    parser.add_argument(
        "--reports-dir",
        default=os.environ.get("DRIFT_REPORTS_DIR", "reports"),
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=int(os.environ.get("DRIFT_SIM_INTERVAL", "30")),
        help="Seconds between drift checks",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.environ.get("DRIFT_SIM_MAX_STEPS", "0")),
        help="0 = run forever",
    )
    parser.add_argument("--once", action="store_true", help="Single iteration (for tests)")
    args = parser.parse_args()

    ref_path = ROOT / args.reference
    if not ref_path.is_file():
        alt = ROOT / "data/raw/synthetic_min.csv"
        if alt.is_file():
            ref_path = alt
        else:
            print(f"ERROR: no reference data at {ref_path}", file=sys.stderr)
            return 1

    reference = load_reference(ref_path)
    reports_dir = ROOT / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_path = ROOT / args.model_path if args.model_path else None

    step = 0
    while True:
        step += 1
        current = make_drifted_batch(reference, step)
        out_path = reports_dir / "incoming" / f"batch_{step:04d}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        current.to_csv(out_path, index=False)

        result = run_once(reference, current, model_path, reports_dir)
        status = "DEGRADED" if result.get("degraded") else "OK"
        print(
            f"[step {step}] drift={status} "
            f"max_ks={result.get('data_drift', {}).get('max_ks')} "
            f"max_psi={result.get('data_drift', {}).get('max_psi')} "
            f"batch={out_path.name}",
            flush=True,
        )

        if args.once or (args.max_steps and step >= args.max_steps):
            break
        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
