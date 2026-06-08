"""FastAPI: scoring, feedback loop, model reload, monitoring dashboard, Web UI."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.models.train_model import ModelBundle, build_features_for_training
from src.monitoring import prometheus_metrics as prom_metrics
from src.monitoring.drift import load_drift_report
from src.monitoring.metrics_store import (
    iter_events_tail,
    log_feedback_event,
    log_score_event,
    new_request_id,
    summarize_events,
)

_bundle: Optional[ModelBundle] = None

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
DRIFT_PATH = Path(os.environ.get("DRIFT_REPORT_PATH", str(REPORTS / "last_drift.json")))
CHAMPION_PATH = Path(
    os.environ.get("CHAMPION_METRICS_PATH", str(REPORTS / "champion_metrics.json"))
)
EVENTS_PATH = Path(os.environ.get("EVENTS_JSONL_PATH", str(REPORTS / "events.jsonl")))

_retrain_task: Optional[asyncio.Task] = None
_retrain_status: Dict[str, Any] = {"state": "idle", "started_at": None, "finished_at": None}


def _default_model_path() -> str:
    for name in ("model_bundle_catboost.pkl", "model_bundle.pkl"):
        p = ROOT / "models" / name
        if p.is_file():
            return str(p)
    return str(ROOT / "models" / "model_bundle_catboost.pkl")


def model_path() -> str:
    return os.environ.get("MODEL_PATH", _default_model_path())


def load_bundle() -> ModelBundle:
    path = model_path()
    if not Path(path).is_file():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    return ModelBundle.load(path)


def reload_bundle() -> None:
    global _bundle
    _bundle = load_bundle()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bundle
    _bundle = load_bundle()
    yield
    _bundle = None


app = FastAPI(
    title="Credit scoring — MLOps API",
    description="Scoring, production feedback, drift report view, model reload, Web UI.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Pydantic models ────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    model_path: str
    model_type: Optional[str] = None


class ScoreRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Raw row as in training CSV")


class ScoreResponse(BaseModel):
    request_id: str
    probability: float = Field(..., ge=0.0, le=1.0)
    model_type: str
    anomaly: bool = False


class FeedbackRequest(BaseModel):
    request_id: str
    y_true: int = Field(..., ge=0, le=1, description="Observed default / bad label")


class FeedbackResponse(BaseModel):
    ok: bool


# ── Auth helper ────────────────────────────────────────────────────────────────


def _require_admin_token(x_admin_token: Optional[str]) -> None:
    expected = os.environ.get("ADMIN_RELOAD_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="ADMIN_RELOAD_TOKEN is not configured on server",
        )
    if not x_admin_token or x_admin_token != expected:
        raise HTTPException(status_code=403, detail="Invalid admin token")


# ── System endpoints ───────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/dashboard")


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_path=model_path(),
        model_type=_bundle.model_type if _bundle else None,
    )


# ── Scoring endpoints ──────────────────────────────────────────────────────────


@app.post("/v1/score", response_model=ScoreResponse, tags=["scoring"])
def score_one(body: ScoreRequest) -> ScoreResponse:
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    rid = new_request_id()
    t0 = time.perf_counter()
    df = pd.DataFrame([body.features])
    target = _bundle.config.get("target_col")
    if target and target in df.columns:
        df = df.drop(columns=[target])
    try:
        df_feat, _ = build_features_for_training(
            df, fit_params=_bundle.preprocessing_params
        )
    except Exception as e:
        prom_metrics.observe_score_error()
        raise HTTPException(status_code=400, detail=f"Feature build failed: {e}") from e
    df_feat = df_feat.reindex(columns=_bundle.feature_cols, fill_value=0)
    proba = float(_bundle.model.predict_proba(df_feat)[:, 1][0])
    ms = (time.perf_counter() - t0) * 1000
    anomaly = proba >= 0.8
    prom_metrics.observe_score_ok(ms / 1000.0, proba)
    log_score_event(
        request_id=rid,
        probability=proba,
        latency_ms=ms,
        model_type=_bundle.model_type,
        path=EVENTS_PATH,
    )
    return ScoreResponse(
        request_id=rid,
        probability=proba,
        model_type=_bundle.model_type,
        anomaly=anomaly,
    )


@app.post("/v1/feedback", response_model=FeedbackResponse, tags=["scoring"])
def feedback(body: FeedbackRequest) -> FeedbackResponse:
    log_feedback_event(request_id=body.request_id, y_true=body.y_true, path=EVENTS_PATH)
    return FeedbackResponse(ok=True)


# ── Monitoring endpoints ───────────────────────────────────────────────────────


@app.get("/v1/metrics/summary", tags=["monitoring"])
def metrics_summary() -> Dict[str, Any]:
    base = summarize_events(path=EVENTS_PATH)
    base["drift"] = load_drift_report(DRIFT_PATH)
    if CHAMPION_PATH.is_file():
        try:
            base["champion"] = json.loads(CHAMPION_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            base["champion"] = None
    else:
        base["champion"] = None
    return base


@app.get("/v1/predictions", tags=["monitoring"])
def recent_predictions(limit: int = 50) -> List[Dict[str, Any]]:
    """Last N scoring events with anomaly flag."""
    rows = []
    for ev in iter_events_tail(path=EVENTS_PATH, max_lines=10000):
        if ev.get("kind") == "score":
            rows.append({
                "request_id": ev.get("request_id"),
                "ts": ev.get("ts"),
                "probability": ev.get("probability"),
                "model_type": ev.get("model_type"),
                "latency_ms": ev.get("latency_ms"),
                "anomaly": (ev.get("probability") or 0) >= 0.8,
            })
    return rows[-limit:][::-1]


@app.get("/v1/drift/report", tags=["monitoring"])
def drift_report() -> Dict[str, Any]:
    report = load_drift_report(DRIFT_PATH)
    if report is None:
        return {"error": "No drift report found. Run scripts/drift_check.py first."}
    return report


# ── Admin endpoints ────────────────────────────────────────────────────────────


@app.post("/internal/reload-model", tags=["admin"])
def reload_model(x_admin_token: Optional[str] = Header(default=None)) -> Dict[str, str]:
    _require_admin_token(x_admin_token)
    try:
        reload_bundle()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"status": "reloaded", "model_path": model_path()}


@app.post("/internal/retrain", tags=["admin"])
def trigger_retrain(
    x_admin_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Trigger model retraining in a background subprocess."""
    _require_admin_token(x_admin_token)
    global _retrain_task, _retrain_status
    if _retrain_status.get("state") == "running":
        return {"status": "already_running", "started_at": _retrain_status.get("started_at")}
    _retrain_status = {"state": "running", "started_at": time.time(), "finished_at": None}

    async def _run() -> None:
        data_path = "data/raw/credit_scoring.csv"
        if not (ROOT / data_path).is_file():
            data_path = "data/raw/synthetic_min.csv"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "retrain_pipeline.py"),
            "--data", str(ROOT / data_path),
            "--model-type", "catboost",
            "--save", str(ROOT / "models" / "model_bundle_catboost.pkl"),
        ]
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if mlflow_uri:
            cmd += ["--mlflow-uri", mlflow_uri, "--mlflow-experiment", "credit_scoring"]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            await proc.wait()
            _retrain_status["state"] = "done" if proc.returncode == 0 else "failed"
        except Exception as exc:
            _retrain_status["state"] = "failed"
            _retrain_status["error"] = str(exc)
        finally:
            _retrain_status["finished_at"] = time.time()
            try:
                reload_bundle()
            except Exception:
                pass

    _retrain_task = asyncio.create_task(_run())
    return {"status": "started", "started_at": _retrain_status["started_at"]}


@app.get("/internal/retrain/status", tags=["admin"])
def retrain_status() -> Dict[str, Any]:
    return _retrain_status


# ── Prometheus metrics endpoint ────────────────────────────────────────────────


@app.get("/metrics", tags=["monitoring"])
def prometheus_metrics() -> Response:
    """Prometheus scrape endpoint."""
    body = prom_metrics.metrics_payload(DRIFT_PATH)
    return Response(
        content=body,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# ── Web UI pages ───────────────────────────────────────────────────────────────


def _load_mlflow_experiments() -> List[Dict[str, Any]]:
    """Try to fetch recent MLflow runs from tracking server."""
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_uri)
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        runs = []
        for exp in experiments[:3]:
            exp_runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=10,
                order_by=["start_time DESC"],
            )
            for r in exp_runs:
                runs.append({
                    "run_id": r.info.run_id[:8],
                    "experiment": exp.name,
                    "status": r.info.status,
                    "start_time": r.info.start_time,
                    "metrics": {k: round(v, 4) for k, v in r.data.metrics.items()},
                    "params": {k: v for k, v in r.data.params.items()
                               if k in ("model_type",)},
                })
        return runs
    except Exception:
        return []


@app.get("/dashboard", response_class=HTMLResponse, tags=["ui"])
def dashboard(request: Request) -> Any:
    summary = summarize_events(path=EVENTS_PATH)
    drift = load_drift_report(DRIFT_PATH)
    champion = None
    if CHAMPION_PATH.is_file():
        try:
            champion = json.loads(CHAMPION_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    predictions = []
    for ev in iter_events_tail(path=EVENTS_PATH, max_lines=2000):
        if ev.get("kind") == "score":
            predictions.append({
                "request_id": str(ev.get("request_id", ""))[:12],
                "ts": ev.get("ts"),
                "probability": round(float(ev.get("probability", 0)), 4),
                "model_type": ev.get("model_type", ""),
                "latency_ms": round(float(ev.get("latency_ms", 0)), 1),
                "anomaly": (ev.get("probability") or 0) >= 0.8,
            })
    predictions = predictions[-100:][::-1]
    drift_alert = bool(drift and drift.get("degraded"))
    offline_auc = None
    if champion and champion.get("models"):
        first = next(iter(champion["models"].values()), {})
        offline_auc = first.get("validation_roc_auc")
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "summary": summary,
            "drift": drift,
            "champion": champion,
            "predictions": predictions,
            "drift_alert": drift_alert,
            "retrain_status": _retrain_status,
            "model_type": _bundle.model_type if _bundle else "unknown",
            "offline_auc": offline_auc,
            "prometheus_url": os.environ.get("PROMETHEUS_UI_URL", "http://localhost:9090"),
            "grafana_url": os.environ.get("GRAFANA_UI_URL", "http://localhost:3000"),
        },
    )


@app.get("/ui/inference", response_class=HTMLResponse, tags=["ui"])
def ui_inference(request: Request) -> Any:
    return templates.TemplateResponse(
        request,
        "inference.html",
        {"model_type": _bundle.model_type if _bundle else "unknown"},
    )


@app.get("/ui/experiments", response_class=HTMLResponse, tags=["ui"])
def ui_experiments(request: Request) -> Any:
    runs = _load_mlflow_experiments()
    champion = None
    if CHAMPION_PATH.is_file():
        try:
            champion = json.loads(CHAMPION_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    mlflow_ui = os.environ.get(
        "MLFLOW_UI_URL",
        os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"),
    )
    return templates.TemplateResponse(
        request,
        "experiments.html",
        {
            "runs": runs,
            "champion": champion,
            "mlflow_url": mlflow_ui,
        },
    )
