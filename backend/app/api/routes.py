import asyncio
import logging
import re
from datetime import datetime, timedelta

import os
import mlflow

mlflow.set_tracking_uri("sqlite:///" + os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mlflow.db")))

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from typing import Dict, Any

from app.db.database import get_db
from app.models.tables import Alert, Forecast, Snapshot
from app.schemas.snapshot import SnapshotCreate
from app.services.forecast import MLForecastService
from app.services.risk import compute_risk
from pydantic import BaseModel

class LoadTestPayload(BaseModel):
    metrics: Dict[str, float]
    tags: Dict[str, str]

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/snapshots")
def create_snapshot(payload: SnapshotCreate, db: Session = Depends(get_db)):
    row = Snapshot(**payload.model_dump())
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.get("/snapshots/latest/{hospital_id}")
def latest_snapshot(hospital_id: str, db: Session = Depends(get_db)):
    row = (
        db.query(Snapshot)
        .filter(Snapshot.hospital_id == hospital_id)
        .order_by(Snapshot.timestamp.desc())
        .first()
    )
    return row


@router.post("/forecast/{hospital_id}")
async def run_forecast(
    hospital_id: str,
    model: str | None = None,
    db: Session = Depends(get_db),
):
    from app.core.config import settings

    rows = (
        db.query(Snapshot)
        .filter(Snapshot.hospital_id == hospital_id)
        .order_by(Snapshot.timestamp.asc())
        .all()
    )

    if not rows:
        return {"message": "No data available for forecasting."}

    df = pd.DataFrame(
        [
            {
                "timestamp": r.timestamp,
                "icu_capacity": r.icu_capacity,
                "icu_occupied": r.icu_occupied,
            }
            for r in rows
        ]
    )

    latest_capacity = int(df["icu_capacity"].iloc[-1])
    model_preference = model or settings.FORECAST_MODEL

    service = MLForecastService(
        model_preference=model_preference,
        horizon=settings.FORECAST_HORIZON,
        frequency=settings.FORECAST_FREQUENCY,
        yellow_threshold=settings.ALERT_OCCUPANCY_YELLOW,
        red_threshold=settings.ALERT_OCCUPANCY_RED,
    )

    # ML inference (model fit + predict + MC CI) is CPU-bound and takes 100–300 ms.
    # Running it in a thread pool keeps the event loop free to handle other requests
    # concurrently instead of blocking them for the full inference duration.
    loop = asyncio.get_running_loop()
    try:
        forecast_steps = await loop.run_in_executor(
            None, service.generate_forecast, df, latest_capacity
        )
    except ValueError as exc:
        return {"message": str(exc), "hospital_id": hospital_id}

    # Delete old forecast + alert rows for this hospital before inserting new ones
    print(">>> DELETING OLD DATA FOR", hospital_id)
    db.query(Forecast).filter(Forecast.hospital_id == hospital_id).delete()
    db.query(Alert).filter(Alert.hospital_id == hospital_id).delete()
    db.commit()

    created = []
    for step in forecast_steps:
        forecast_row = Forecast(
            hospital_id=hospital_id,
            forecast_time=step.forecast_time,
            predicted_icu_occupied=step.predicted_icu_occupied,
            yhat_lower=step.yhat_lower,
            yhat_upper=step.yhat_upper,
            risk_level=step.risk_level,
            model_name=step.model_name,
            breach_prob=step.breach_prob,
        )
        db.add(forecast_row)

        created.append(
            {
                "forecast_time": step.forecast_time.isoformat(),
                "predicted_icu_occupied": step.predicted_icu_occupied,
                "yhat_lower": step.yhat_lower,
                "yhat_upper": step.yhat_upper,
                "risk_level": step.risk_level,
                "model_name": step.model_name,
                "breach_prob": step.breach_prob,
            }
        )

        if step.risk_level in {"YELLOW", "RED"}:
            alert = Alert(
                hospital_id=hospital_id,
                created_at=step.forecast_time,
                risk_level=step.risk_level,
                message=f"Predicted ICU occupancy risk {step.risk_level} at {step.forecast_time.isoformat()}",
                breach_prob=step.breach_prob,
            )
            db.add(alert)

    db.commit()
    return {
        "hospital_id": hospital_id,
        "model": forecast_steps[0].model_name if forecast_steps else None,
        "horizon": len(forecast_steps),
        "forecasts": created,
    }


@router.get("/forecasts/{hospital_id}")
def get_forecasts(hospital_id: str, db: Session = Depends(get_db)):
    rows = (
        db.query(Forecast)
        .filter(Forecast.hospital_id == hospital_id)
        .order_by(Forecast.forecast_time.asc())
        .all()
    )
    return rows


@router.get("/hospitals")
def get_hospitals(db: Session = Depends(get_db)):
    rows = db.query(Snapshot.hospital_id).distinct().order_by(Snapshot.hospital_id).all()
    return [r[0] for r in rows]


@router.get("/alerts")
def get_all_alerts(limit: int = 50, db: Session = Depends(get_db)):
    rows = (
        db.query(Alert)
        .order_by(Alert.created_at.desc())
        .limit(limit)
        .all()
    )
    return rows


@router.get("/alerts/{hospital_id}")
def get_alerts(hospital_id: str, db: Session = Depends(get_db)):
    rows = (
        db.query(Alert)
        .filter(Alert.hospital_id == hospital_id)
        .order_by(Alert.created_at.desc())
        .all()
    )
    return rows


def _safe_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


_HOSPITAL_METRIC_RE = re.compile(r"^(\d{6})_(p50_ms|p95_ms|p99_ms|failure_rate|rps)$")
_LOAD_TEST_EXPERIMENT = "hospital-load-test"


@router.get("/load-test/latest")
def get_load_test_latest():
    try:
        runs = mlflow.search_runs(
            experiment_names=[_LOAD_TEST_EXPERIMENT],
            max_results=1,
            order_by=["start_time DESC"],
        )
    except Exception as exc:
        logger.warning("MLflow search_runs failed: %s", exc)
        raise HTTPException(status_code=404, detail="no runs found")

    if runs.empty:
        raise HTTPException(status_code=404, detail="no runs found")

    run = runs.iloc[0]
    tags = {k[len("tags."):]: v for k, v in run.items() if k.startswith("tags.")}
    metrics = {k[len("metrics."):]: v for k, v in run.items() if k.startswith("metrics.")}

    hospitals: dict[str, dict] = {}
    for key, val in metrics.items():
        m = _HOSPITAL_METRIC_RE.match(key)
        if m:
            hid, field = m.group(1), m.group(2)
            hospitals.setdefault(hid, {"hospital_id": hid})[field] = round(float(val), 3)

    return {
        "run_id":               run.get("run_id", ""),
        "timestamp":            tags.get("timestamp", ""),
        "num_hospitals":        _safe_int(tags.get("num_hospitals", 0)),
        "run_duration_s":       _safe_float(tags.get("run_duration_s", 0)),
        "overall_p95_ms":       _safe_float(metrics.get("overall_p95_ms", 0)),
        "overall_failure_rate": _safe_float(metrics.get("overall_failure_rate", 0)),
        "overall_rps":          _safe_float(metrics.get("overall_rps", 0)),
        "hospitals":            sorted(hospitals.values(), key=lambda h: h["hospital_id"]),
    }


@router.post("/load-test/runs")
def create_load_test_run(payload: LoadTestPayload):
    mlflow.set_experiment(_LOAD_TEST_EXPERIMENT)
    with mlflow.start_run():
        mlflow.log_metrics(payload.metrics)
        mlflow.set_tags(payload.tags)
    return {"status": "ok"}


@router.get("/load-test/runs")
def get_load_test_runs():
    try:
        runs = mlflow.search_runs(
            experiment_names=[_LOAD_TEST_EXPERIMENT],
            max_results=20,
            order_by=["start_time DESC"],
        )
    except Exception as exc:
        logger.warning("MLflow search_runs failed: %s", exc)
        return []

    if runs.empty:
        return []

    result = []
    for _, run in runs.iterrows():
        tags = {k[len("tags."):]: v for k, v in run.items() if k.startswith("tags.")}
        metrics = {k[len("metrics."):]: v for k, v in run.items() if k.startswith("metrics.")}
        result.append({
            "run_id":               run.get("run_id", ""),
            "timestamp":            tags.get("timestamp", str(run.get("start_time", ""))),
            "num_hospitals":        _safe_int(tags.get("num_hospitals", 0)),
            "run_duration_s":       _safe_float(tags.get("run_duration_s", 0)),
            "overall_p95_ms":       _safe_float(metrics.get("overall_p95_ms", 0)),
            "overall_failure_rate": _safe_float(metrics.get("overall_failure_rate", 0)),
            "overall_rps":          _safe_float(metrics.get("overall_rps", 0)),
        })
    return result