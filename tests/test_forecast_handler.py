"""Moto integration tests for lambdas/forecast/handler.lambda_handler.

_select_model is patched to return a fast BaselineForecaster, avoiding the
slow Prophet/SARIMA fitting that would make these tests take 30+ seconds.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch

import boto3
import pandas as pd
import pytest
from moto import mock_aws

SNAPSHOTS_TABLE = "test-snapshots"
FORECASTS_TABLE = "test-forecasts"
ALERTS_TABLE    = "test-alerts"
DATA_BUCKET     = "test-data-bucket"

os.environ.setdefault("SNAPSHOTS_TABLE",              SNAPSHOTS_TABLE)
os.environ.setdefault("FORECASTS_TABLE",              FORECASTS_TABLE)
os.environ.setdefault("ALERTS_TABLE",                 ALERTS_TABLE)
os.environ.setdefault("DATA_BUCKET",                  DATA_BUCKET)
os.environ.setdefault("POWERTOOLS_SERVICE_NAME",      "test")
os.environ.setdefault("POWERTOOLS_METRICS_NAMESPACE", "TestNS")
os.environ.setdefault("POWERTOOLS_TRACE_DISABLED",    "true")
os.environ.setdefault("AWS_DEFAULT_REGION",           "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID",            "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY",        "testing")
os.environ.setdefault("ALERT_OCCUPANCY_YELLOW",       "0.75")
os.environ.setdefault("ALERT_OCCUPANCY_RED",          "0.90")
os.environ.setdefault("BREACH_PROB_THRESHOLD",        "0.60")
os.environ.setdefault("FORECAST_HORIZON_WEEKS",       "4")

from ml.models.baseline import BaselineForecaster       # noqa: E402
from lambdas.forecast.handler import lambda_handler     # noqa: E402


class _FakeContext:
    function_name          = "test-forecast"
    memory_limit_in_mb     = 2048
    invoked_function_arn   = "arn:aws:lambda:us-east-1:000000000000:function:test-forecast"
    aws_request_id         = "test-req-id"

_CTX = _FakeContext()


def _make_tables(dynamodb):
    for name in [SNAPSHOTS_TABLE, FORECASTS_TABLE, ALERTS_TABLE]:
        dynamodb.create_table(
            TableName=name,
            KeySchema=[
                {"AttributeName": "pk", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )


def _seed_snapshots(dynamodb, hospital_id: str, n: int = 12) -> pd.DataFrame:
    """Insert n weekly snapshot items and return them as a DataFrame for model fitting."""
    table = dynamodb.Table(SNAPSHOTS_TABLE)
    base  = datetime(2024, 1, 7, tzinfo=timezone.utc)
    rows  = []
    with table.batch_writer() as batch:
        for i in range(n):
            ts       = base + timedelta(weeks=i)
            occupied = 70 + (i % 5)
            capacity = 100
            batch.put_item(Item={
                "pk":              f"HOSPITAL#{hospital_id}",
                "sk":              f"SNAPSHOT#{ts.isoformat()}",
                "hospital_id":     hospital_id,
                "timestamp":       ts.isoformat(),
                "icu_capacity":    Decimal(str(capacity)),
                "icu_occupied":    Decimal(str(occupied)),
                "admissions":      Decimal("5"),
                "discharges":      Decimal("4"),
                "transfers":       Decimal("1"),
                "ed_arrivals":     Decimal("10"),
                "staffing_level":  Decimal("0.90"),
                "occupancy_ratio": Decimal(str(round(occupied / capacity, 4))),
            })
            rows.append({"timestamp": ts, "icu_occupied": float(occupied), "icu_capacity": float(capacity)})
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def _fitted_baseline(df: pd.DataFrame):
    m = BaselineForecaster(target_col="icu_occupied")
    m.fit(df)
    return m, "baseline"


# ── test 1: end-to-end success ────────────────────────────────────────────────

@mock_aws
def test_forecast_success_status():
    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    s3  = boto3.client("s3",         region_name="us-east-1")
    _make_tables(ddb)
    s3.create_bucket(Bucket=DATA_BUCKET)
    df = _seed_snapshots(ddb, "H001")

    with patch("lambdas.forecast.handler._select_model", return_value=_fitted_baseline(df)):
        result = lambda_handler({"hospital_id": "H001"}, _CTX)

    assert result["status"] == "complete"
    h = next(r for r in result["hospitals"] if r["hospital_id"] == "H001")
    assert h["status"] == "success"
    assert h["model"] == "baseline"


# ── test 2: forecasts written with required fields ────────────────────────────

@mock_aws
def test_forecast_items_written_to_dynamodb():
    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    s3  = boto3.client("s3",         region_name="us-east-1")
    _make_tables(ddb)
    s3.create_bucket(Bucket=DATA_BUCKET)
    df = _seed_snapshots(ddb, "H001")

    with patch("lambdas.forecast.handler._select_model", return_value=_fitted_baseline(df)):
        lambda_handler({"hospital_id": "H001"}, _CTX)

    table = ddb.Table(FORECASTS_TABLE)
    items = table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key("pk").eq("HOSPITAL#H001")
    )["Items"]

    assert len(items) == 4  # FORECAST_HORIZON_WEEKS=4
    for item in items:
        assert "predicted_icu_occupied" in item
        assert "yhat_lower" in item
        assert "yhat_upper" in item
        assert "risk_level" in item
        assert item["risk_level"] in ("GREEN", "YELLOW", "RED")
        assert "breach_prob" in item


# ── test 3: insufficient data returns status ──────────────────────────────────

@mock_aws
def test_insufficient_data_returns_status():
    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    s3  = boto3.client("s3",         region_name="us-east-1")
    _make_tables(ddb)
    s3.create_bucket(Bucket=DATA_BUCKET)
    _seed_snapshots(ddb, "H002", n=2)  # only 2 rows — below the 4-row minimum

    with patch("lambdas.forecast.handler._select_model", return_value=(None, "none")):
        result = lambda_handler({"hospital_id": "H002"}, _CTX)

    h = next(r for r in result["hospitals"] if r["hospital_id"] == "H002")
    assert h["status"] == "insufficient_data"


# ── test 4: previous forecasts are cleared before writing new ones ────────────

@mock_aws
def test_old_forecasts_are_replaced():
    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    s3  = boto3.client("s3",         region_name="us-east-1")
    _make_tables(ddb)
    s3.create_bucket(Bucket=DATA_BUCKET)
    df = _seed_snapshots(ddb, "H001")

    with patch("lambdas.forecast.handler._select_model", return_value=_fitted_baseline(df)):
        lambda_handler({"hospital_id": "H001"}, _CTX)
        lambda_handler({"hospital_id": "H001"}, _CTX)

    table = ddb.Table(FORECASTS_TABLE)
    items = table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key("pk").eq("HOSPITAL#H001")
    )["Items"]
    # Two runs each write 4 forecasts, but old run is cleared → only 4 remain
    assert len(items) == 4


# ── test 5: S3 artifact load path (joblib bundle is used when found) ──────────

@mock_aws
def test_s3_artifact_loaded_when_present():
    import io
    import joblib

    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    s3  = boto3.client("s3",         region_name="us-east-1")
    _make_tables(ddb)
    s3.create_bucket(Bucket=DATA_BUCKET)
    df = _seed_snapshots(ddb, "H001")

    # Serialize a pre-fitted baseline model as a joblib bundle into S3
    model, model_name = _fitted_baseline(df)
    bundle = {"model": model, "model_name": model_name}
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    buf.seek(0)
    s3.put_object(Bucket=DATA_BUCKET, Key="artifacts/models/H001_best.joblib", Body=buf.read())

    # Do NOT patch _select_model — let the handler load from S3 naturally
    result = lambda_handler({"hospital_id": "H001"}, _CTX)

    h = next(r for r in result["hospitals"] if r["hospital_id"] == "H001")
    assert h["status"] == "success"
    assert h["model"] == "baseline"


# ── SQS trigger path ──────────────────────────────────────────────────────────

def _sqs_event(hospital_id: str) -> dict:
    """Minimal SQS Records envelope — mirrors what AWS delivers at BatchSize=1."""
    import json as _json
    return {"Records": [{"body": _json.dumps({"hospital_id": hospital_id})}]}


@mock_aws
def test_sqs_trigger_runs_forecast():
    """SQS envelope routes to the ML pipeline and writes results to DynamoDB."""
    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    s3  = boto3.client("s3",         region_name="us-east-1")
    _make_tables(ddb)
    s3.create_bucket(Bucket=DATA_BUCKET)
    df = _seed_snapshots(ddb, "H001")

    with patch("lambdas.forecast.handler._select_model", return_value=_fitted_baseline(df)):
        result = lambda_handler(_sqs_event("H001"), _CTX)

    assert result["status"] == "complete"
    h = next(r for r in result["hospitals"] if r["hospital_id"] == "H001")
    assert h["status"] == "success"


@mock_aws
def test_sqs_missing_hospital_id_is_skipped():
    """Record without hospital_id emits a warning and returns an empty hospital list."""
    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    s3  = boto3.client("s3",         region_name="us-east-1")
    _make_tables(ddb)
    s3.create_bucket(Bucket=DATA_BUCKET)

    import json as _json
    bad_event = {"Records": [{"body": _json.dumps({"not_a_hospital": "oops"})}]}
    result = lambda_handler(bad_event, _CTX)

    assert result["status"] == "complete"
    assert result["hospitals"] == []
