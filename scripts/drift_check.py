#!/usr/bin/env python3
"""Full drift report: data (KS+PSI), target (prevalence), concept (corr + score KS)."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.models.train_model import ModelBundle
from src.monitoring.drift import compute_full_drift_report, save_drift_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Data / target / concept drift vs reference CSV",
    )
    parser.add_argument(
        "--reference",
        default="data/raw/credit_scoring.csv",
        help="Reference (e.g. training) CSV",
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Current production or batch CSV",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional ModelBundle .pkl for score-distribution concept drift",
    )
    parser.add_argument("--ks-threshold", type=float, default=0.2)
    parser.add_argument("--psi-threshold", type=float, default=0.25)
    parser.add_argument("--report-path", default="reports/last_drift.json")
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit 1 if any dimension is in alert state",
    )
    args = parser.parse_args()

    ref = pd.read_csv(args.reference)
    cur = pd.read_csv(args.current)
    bundle = ModelBundle.load(args.model_path) if args.model_path else None
    report = compute_full_drift_report(
        ref,
        cur,
        bundle=bundle,
        ks_alert_threshold=args.ks_threshold,
        psi_alert_threshold=args.psi_threshold,
    )
    save_drift_report(report, Path(args.report_path))
    print(report.to_dict())
    if args.fail_on_drift and report.degraded:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
