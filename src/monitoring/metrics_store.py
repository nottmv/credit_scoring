"""Append-only JSONL log of scoring + feedback for dashboard and quality checks."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

_lock = threading.Lock()


def _default_path() -> Path:
    return Path(__file__).resolve().parents[2] / "reports" / "events.jsonl"


def append_event(event: Dict[str, Any], path: Optional[Path] = None) -> None:
    path = path or _default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False) + "\n"
    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def log_score_event(
    *,
    request_id: str,
    probability: float,
    latency_ms: float,
    model_type: str,
    path: Optional[Path] = None,
) -> None:
    append_event(
        {
            "ts": time.time(),
            "kind": "score",
            "request_id": request_id,
            "probability": probability,
            "latency_ms": round(latency_ms, 3),
            "model_type": model_type,
        },
        path=path,
    )


def log_feedback_event(
    *,
    request_id: str,
    y_true: int,
    path: Optional[Path] = None,
) -> None:
    append_event(
        {
            "ts": time.time(),
            "kind": "feedback",
            "request_id": request_id,
            "y_true": int(y_true),
        },
        path=path,
    )


def iter_events_tail(
    path: Optional[Path] = None, max_lines: int = 8000
) -> Iterator[Dict[str, Any]]:
    path = path or _default_path()
    lines: List[str] = []
    if path.is_file():
        with _lock:
            with path.open("r", encoding="utf-8") as f:
                lines = f.readlines()[-max_lines:]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def summarize_events(path: Optional[Path] = None, max_lines: int = 8000) -> Dict[str, Any]:
    scores: List[Dict[str, Any]] = []
    feedback: Dict[str, int] = {}
    for ev in iter_events_tail(path=path, max_lines=max_lines):
        if ev.get("kind") == "score":
            scores.append(ev)
        elif ev.get("kind") == "feedback":
            rid = ev.get("request_id")
            if rid is not None:
                feedback[str(rid)] = int(ev["y_true"])

    n_score = len(scores)
    latencies = [float(s["latency_ms"]) for s in scores if "latency_ms" in s]
    probs = [float(s["probability"]) for s in scores if "probability" in s]

    def pct(xs: List[float], q: float) -> Optional[float]:
        if not xs:
            return None
        xs = sorted(xs)
        idx = min(len(xs) - 1, int(q * (len(xs) - 1)))
        return round(xs[idx], 3)

    # production AUC from scored rows that got feedback
    paired_p: List[float] = []
    paired_y: List[int] = []
    score_by_id = {str(s["request_id"]): s for s in scores if "request_id" in s}
    for rid, y in feedback.items():
        if rid in score_by_id:
            paired_p.append(float(score_by_id[rid]["probability"]))
            paired_y.append(y)

    prod_auc = None
    if len(set(paired_y)) > 1 and len(paired_y) >= 2:
        from sklearn.metrics import roc_auc_score

        prod_auc = float(roc_auc_score(paired_y, paired_p))

    return {
        "n_scores": n_score,
        "n_feedback": len(feedback),
        "n_pairs_for_auc": len(paired_y),
        "production_roc_auc": prod_auc,
        "latency_p50_ms": pct(latencies, 0.5),
        "latency_p95_ms": pct(latencies, 0.95),
        "score_rate_high_risk": (
            round(sum(1 for p in probs if p >= 0.5) / len(probs), 4) if probs else None
        ),
        "mean_probability": round(sum(probs) / len(probs), 4) if probs else None,
    }


def new_request_id() -> str:
    return str(uuid.uuid4())
