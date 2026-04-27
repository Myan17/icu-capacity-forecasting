"""API Lambda — serves the React dashboard via API Gateway HTTP API.

Uses Mangum to wrap FastAPI so the same route definitions work both
locally (uvicorn) and in Lambda. This also means the existing Postman/curl
test suite continues to work against both the EC2 and serverless stacks.

Observability: Lambda Powertools structured logging + X-Ray tracing.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import boto3
from aws_lambda_powertools import Logger, Metrics, Tracer
from aws_lambda_powertools.metrics import MetricUnit
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

from shared.dynamo import (
    query_alerts,
    query_forecasts,
    query_latest_snapshot,
    query_recent_snapshots,
    snapshot_pk,
    snapshots_table,
    forecasts_table,
    alerts_table,
)
from shared.models import SnapshotItem

logger  = Logger(service="hospital-api")
tracer  = Tracer(service="hospital-api")
metrics = Metrics(namespace="HospitalForecasting", service="hospital-api")

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="ICU Capacity Forecast API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Helper: serialise Decimal → float for JSON responses ─────────────────────

def _clean(obj):
    if isinstance(obj, list):
        return [_clean(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, Decimal):
        return float(obj)
    return obj


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
@tracer.capture_method
def health():
    metrics.add_metric(name="HealthChecks", unit=MetricUnit.Count, value=1)
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/snapshots/latest/{hospital_id}")
@tracer.capture_method
def latest_snapshot(hospital_id: str):
    item = query_latest_snapshot(hospital_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"No snapshots found for {hospital_id}")
    return _clean(item)


@app.get("/snapshots/{hospital_id}")
@tracer.capture_method
def get_snapshots(hospital_id: str, limit: int = 52):
    items = query_recent_snapshots(hospital_id, limit=min(limit, 104))
    return _clean(items)


@app.get("/forecasts/{hospital_id}")
@tracer.capture_method
def get_forecasts(hospital_id: str):
    items = query_forecasts(hospital_id)
    metrics.add_metric(name="ForecastReads", unit=MetricUnit.Count, value=1)
    return _clean(items)


@app.get("/alerts/{hospital_id}")
@tracer.capture_method
def get_alerts(hospital_id: str, limit: int = 50):
    items = query_alerts(hospital_id, limit=min(limit, 200))
    return _clean(items)


@app.post("/forecast/{hospital_id}", status_code=202)
@tracer.capture_method
def trigger_forecast(hospital_id: str):
    """Enqueue an async forecast job. Returns 202 immediately; ForecastFunction
    consumes from SQS, runs the ML pipeline, and writes results to DynamoDB.
    Poll GET /forecasts/{hospital_id} to retrieve results when ready."""
    sqs = boto3.client("sqs")
    queue_url = os.environ["FORECAST_QUEUE_URL"]
    try:
        sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps({"hospital_id": hospital_id}),
        )
        metrics.add_metric(name="ForecastTriggers", unit=MetricUnit.Count, value=1)
        logger.info("Forecast queued", extra={"hospital_id": hospital_id})
        return {"status": "queued", "hospital_id": hospital_id}
    except Exception as exc:
        logger.error("Forecast enqueue failed", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail=str(exc))


# ── Mangum adapter (API Gateway HTTP API ↔ FastAPI) ───────────────────────────

@logger.inject_lambda_context(log_event=False)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: Any):
    handler = Mangum(app, lifespan="off")
    return handler(event, context)
