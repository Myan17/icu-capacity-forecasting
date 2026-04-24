"""Forecast Lambda — Phase 3 core implementation.

Flow:
  1. Load snapshot history for hospital from DynamoDB (up to 52 weeks)
  2. Download model artifacts from S3 (if pre-trained artifacts exist)
     else run BestModelSelector to pick and fit the best model
  3. Generate forecast with confidence intervals (yhat_lower / yhat_upper)
  4. Compute probability-of-breach for each step
  5. Write forecast items to DynamoDB (replacing previous run)
  6. Write probabilistic alerts (deduped by breach window)

Runs inside a Docker container image (ECR) to support Prophet + statsmodels.
Observability: Lambda Powertools structured logging + custom CloudWatch metrics.

Note on serialization: model artifacts are loaded from a private S3 bucket with
server-side encryption and block-public-access enabled (see template.yaml).
The serialization format (joblib/pickle) is appropriate for scikit-learn-compatible
objects; bucket access is restricted to the ForecastFunction IAM role only.
"""
from __future__ import annotations

import io
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import boto3
import joblib          # safer ML serialization than raw pickle
import numpy as np
import pandas as pd
from aws_lambda_powertools import Logger, Metrics, Tracer
from aws_lambda_powertools.metrics import MetricUnit
from boto3.dynamodb.conditions import Key
from scipy.stats import norm

# ML pipeline (bundled in container image)
import sys
sys.path.insert(0, "/var/task")

from ml.config import DEFAULT_CONFIG
from ml.models.baseline import BaselineForecaster
from ml.models.prophet_model import ProphetForecaster
from ml.models.sarima_model import SarimaForecaster
from ml.model_selection.selector import BestModelSelector

from shared.dynamo import (
    alert_dedupe_key,
    alert_exists,
    alert_sk,
    alerts_table,
    forecast_sk,
    forecasts_table,
    query_recent_snapshots,
    snapshot_pk,
    snapshots_table,
)
from shared.models import ForecastEvent
from lambdas.forecast.forecast_utils import breach_prob, risk_label, mc_ci

logger  = Logger(service="hospital-forecast")
tracer  = Tracer(service="hospital-forecast")
metrics = Metrics(namespace="HospitalForecasting", service="hospital-forecast")

S3 = boto3.client("s3")
DATA_BUCKET      = os.environ.get("DATA_BUCKET", "")
YELLOW_THRESHOLD = float(os.environ.get("ALERT_OCCUPANCY_YELLOW", 0.75))
RED_THRESHOLD    = float(os.environ.get("ALERT_OCCUPANCY_RED", 0.90))
BREACH_THRESHOLD = float(os.environ.get("BREACH_PROB_THRESHOLD", 0.60))
HORIZON          = int(os.environ.get("FORECAST_HORIZON_WEEKS", 4))
MC_SAMPLES       = 500   # Monte Carlo draws for confidence intervals on non-Prophet models


# ── Entry point ───────────────────────────────────────────────────────────────

@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: Any) -> dict:
    payload = ForecastEvent(**event) if event else ForecastEvent()
    hospital_ids = _resolve_hospital_ids(payload.hospital_id)

    results = []
    for hid in hospital_ids:
        try:
            result = _run_forecast_for_hospital(hid)
            results.append(result)
        except Exception as exc:
            logger.error("Forecast failed", extra={"hospital_id": hid, "error": str(exc)})
            results.append({"hospital_id": hid, "status": "error", "error": str(exc)})

    return {"status": "complete", "hospitals": results}


def _resolve_hospital_ids(hospital_id: str | None) -> list[str]:
    if hospital_id:
        return [hospital_id]
    table = snapshots_table()
    resp = table.scan(ProjectionExpression="pk", Select="SPECIFIC_ATTRIBUTES")
    pks = {item["pk"] for item in resp.get("Items", [])}
    return [pk.replace("HOSPITAL#", "") for pk in pks if pk.startswith("HOSPITAL#")]


# ── Per-hospital pipeline ─────────────────────────────────────────────────────

@tracer.capture_method
def _run_forecast_for_hospital(hospital_id: str) -> dict:
    logger.info("Starting forecast", extra={"hospital_id": hospital_id})

    raw_items = query_recent_snapshots(hospital_id, limit=52)
    if len(raw_items) < 4:
        return {"hospital_id": hospital_id, "status": "insufficient_data", "rows": len(raw_items)}

    df = _items_to_dataframe(raw_items)
    latest_capacity = int(df["icu_capacity"].iloc[-1])

    model, model_name = _select_model(hospital_id, df)
    steps = _generate_forecast(model, model_name, df)

    run_id = uuid.uuid4().hex[:8]
    _write_forecasts(hospital_id, run_id, model_name, steps, latest_capacity)
    _write_probabilistic_alerts(hospital_id, steps, latest_capacity)

    metrics.add_metric(name="ForecastsGenerated", unit=MetricUnit.Count, value=1)
    logger.info("Forecast complete", extra={"hospital_id": hospital_id, "model": model_name, "run_id": run_id})
    return {"hospital_id": hospital_id, "status": "success", "model": model_name, "run_id": run_id}


def _items_to_dataframe(items: list[dict]) -> pd.DataFrame:
    rows = [
        {
            "timestamp":    pd.to_datetime(item["timestamp"]),
            "icu_occupied": float(item.get("icu_occupied", 0)),
            "icu_capacity": float(item.get("icu_capacity", 1)),
            "hospital_id":  item.get("hospital_id", ""),
        }
        for item in items
    ]
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


# ── Model selection ───────────────────────────────────────────────────────────

def _select_model(hospital_id: str, df: pd.DataFrame):
    artifact = _load_artifact_from_s3(hospital_id)
    if artifact:
        logger.info("Using pre-trained artifact", extra={"hospital_id": hospital_id, "model": artifact["model_name"]})
        return artifact["model"], artifact["model_name"]

    logger.info("Running BestModelSelector", extra={"hospital_id": hospital_id})
    selector = BestModelSelector(
        metric=DEFAULT_CONFIG.selection_metric,
        horizon=min(DEFAULT_CONFIG.test_size, max(len(df) // 4, 2)),
        timestamp_col="timestamp",
        target_col="icu_occupied",
        frequency="W",
    )
    result = selector.select_best_model(df.copy())
    return result.best_model, result.best_model_name


def _load_artifact_from_s3(hospital_id: str):
    """Load a joblib-serialized model artifact from the private data bucket."""
    if not DATA_BUCKET:
        return None
    key = f"artifacts/models/{hospital_id}_best.joblib"
    try:
        obj = S3.get_object(Bucket=DATA_BUCKET, Key=key)
        buf = io.BytesIO(obj["Body"].read())
        return joblib.load(buf)
    except Exception:
        return None


# ── Forecast + intervals ──────────────────────────────────────────────────────

def _generate_forecast(model, model_name: str, df: pd.DataFrame) -> list[dict]:
    if isinstance(model, ProphetForecaster):
        return _prophet_forecast(model, df)
    return _mc_forecast(model, model_name, df)


def _prophet_forecast(model: ProphetForecaster, df: pd.DataFrame) -> list[dict]:
    raw = model.predict(HORIZON)
    now = datetime.now(timezone.utc)
    steps = []
    for i in range(HORIZON):
        if isinstance(raw, pd.DataFrame) and len(raw) > i:
            yhat       = float(raw.iloc[i].get("yhat",       raw.iloc[i].iloc[0]))
            yhat_lower = float(raw.iloc[i].get("yhat_lower", yhat * 0.9))
            yhat_upper = float(raw.iloc[i].get("yhat_upper", yhat * 1.1))
        else:
            val = float(raw[i]) if hasattr(raw, "__getitem__") else float(raw)
            yhat = yhat_lower = yhat_upper = val
        steps.append({
            "forecast_time": now + timedelta(weeks=i + 1),
            "yhat":          yhat,
            "yhat_lower":    yhat_lower,
            "yhat_upper":    yhat_upper,
        })
    return steps


def _mc_forecast(model, model_name: str, df: pd.DataFrame) -> list[dict]:
    """Monte Carlo bootstrap for confidence intervals on Baseline/SARIMA."""
    series = df["icu_occupied"].values.astype(float)
    preds = np.array(model.predict(HORIZON), dtype=float)

    split = max(len(series) - HORIZON, int(len(series) * 0.8))
    train_df = df.iloc[:split].copy()
    actuals = series[split:]
    residuals = _compute_residuals(model_name, train_df, len(actuals), actuals)

    now = datetime.now(timezone.utc)
    steps = []
    for i, yhat in enumerate(preds):
        lo, hi = mc_ci(residuals, float(yhat), seed=42 + i)
        steps.append({
            "forecast_time": now + timedelta(weeks=i + 1),
            "yhat":          float(yhat),
            "yhat_lower":    lo,
            "yhat_upper":    hi,
        })
    return steps


def _compute_residuals(model_name: str, train_df: pd.DataFrame, n: int, actuals: np.ndarray) -> np.ndarray:
    try:
        if model_name == "baseline":
            m = BaselineForecaster(target_col="icu_occupied")
        elif model_name == "sarima":
            m = SarimaForecaster(target_col="icu_occupied")
        else:
            m = BaselineForecaster(target_col="icu_occupied")
        m.fit(train_df)
        preds = np.array(m.predict(n), dtype=float)
        k = min(len(preds), len(actuals))
        return actuals[:k] - preds[:k]
    except Exception as exc:
        logger.warning("Residual computation failed", extra={"error": str(exc)})
        return np.zeros(1)


# ── Breach probability ────────────────────────────────────────────────────────

def _breach_prob(step: dict, capacity: int) -> float:
    return breach_prob(
        yhat=step["yhat"],
        yhat_lower=step["yhat_lower"],
        yhat_upper=step["yhat_upper"],
        capacity=capacity,
        red_threshold=RED_THRESHOLD,
    )


# ── DynamoDB writes ───────────────────────────────────────────────────────────

def _write_forecasts(
    hospital_id: str, run_id: str, model_name: str,
    steps: list[dict], capacity: int,
) -> None:
    table = forecasts_table()
    # Clear previous forecasts for this hospital
    pk = snapshot_pk(hospital_id)
    old = table.query(KeyConditionExpression=Key("pk").eq(pk)).get("Items", [])
    with table.batch_writer() as batch:
        for item in old:
            batch.delete_item(Key={"pk": item["pk"], "sk": item["sk"]})

    with table.batch_writer() as batch:
        for step in steps:
            ft: datetime = step["forecast_time"]
            yhat = step["yhat"]
            risk = risk_label(yhat, capacity, yellow=YELLOW_THRESHOLD, red=RED_THRESHOLD)
            bp = _breach_prob(step, capacity)

            batch.put_item(Item={
                "pk":                      snapshot_pk(hospital_id),
                "sk":                      forecast_sk(run_id, ft),
                "hospital_id":             hospital_id,
                "run_id":                  run_id,
                "forecast_time":           ft.isoformat(),
                "predicted_icu_occupied":  Decimal(str(round(yhat, 2))),
                "yhat_lower":              Decimal(str(round(step["yhat_lower"], 2))),
                "yhat_upper":              Decimal(str(round(step["yhat_upper"], 2))),
                "risk_level":              risk,
                "model_name":              model_name,
                "breach_prob":             Decimal(str(round(bp, 4))),
                "icu_capacity":            capacity,
            })


def _write_probabilistic_alerts(hospital_id: str, steps: list[dict], capacity: int) -> None:
    table = alerts_table()
    now = datetime.now(timezone.utc)

    for step in steps:
        bp = _breach_prob(step, capacity)
        if bp < BREACH_THRESHOLD:
            continue

        ft: datetime = step["forecast_time"]
        ratio = step["yhat"] / max(capacity, 1)
        risk = "RED" if ratio >= RED_THRESHOLD else "YELLOW"
        eta = round((ft - now).total_seconds() / 3600, 1)
        dedupe = alert_dedupe_key(hospital_id, risk, ft)

        if alert_exists(hospital_id, dedupe):
            continue

        table.put_item(Item={
            "pk":               snapshot_pk(hospital_id),
            "sk":               alert_sk(now),
            "hospital_id":      hospital_id,
            "created_at":       now.isoformat(),
            "risk_level":       risk,
            "message":          (
                f"P(ICU ≥ {int(RED_THRESHOLD*100)}% capacity) = {bp:.0%} "
                f"at {ft.strftime('%Y-%m-%d')} (ETA {eta:.0f}h)"
            ),
            "breach_prob":      Decimal(str(round(bp, 4))),
            "breach_eta_hours": Decimal(str(eta)),
            "dedupe_key":       dedupe,
        })
        metrics.add_metric(name="AlertsRaised", unit=MetricUnit.Count, value=1)
