"""Tests for the new L1/L2 features added to lambdas/forecast/handler.py:
  * Batch invocation (``hospital_ids: [...]``)
  * Priority ordering
  * Retry + baseline fallback when the selector fails
  * Workload-tier metadata in the response payload
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch

import boto3
import pandas as pd
from moto import mock_aws

# Use the same table names as test_forecast_handler.py so we don't conflict
# with its os.environ.setdefault values when the test runner imports both files.
SNAPSHOTS_TABLE = os.environ.setdefault("SNAPSHOTS_TABLE", "test-snapshots")
FORECASTS_TABLE = os.environ.setdefault("FORECASTS_TABLE", "test-forecasts")
ALERTS_TABLE    = os.environ.setdefault("ALERTS_TABLE",    "test-alerts")
DATA_BUCKET     = os.environ.setdefault("DATA_BUCKET",     "test-data-bucket")
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

from ml.models.baseline import BaselineForecaster        # noqa: E402
from lambdas.forecast import handler as fc_handler        # noqa: E402


class _Ctx:
    function_name        = "test-forecast"
    memory_limit_in_mb   = 2048
    invoked_function_arn = "arn:aws:lambda:us-east-1:000000000000:function:test-forecast"
    aws_request_id       = "test-req-id"


_CTX = _Ctx()


def _make_tables(ddb):
    for name in (SNAPSHOTS_TABLE, FORECASTS_TABLE, ALERTS_TABLE):
        ddb.create_table(
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


def _seed(ddb, hospital_id: str, n: int = 12, occupied_base: int = 70) -> pd.DataFrame:
    table = ddb.Table(SNAPSHOTS_TABLE)
    base = datetime(2024, 1, 7, tzinfo=timezone.utc)
    rows = []
    with table.batch_writer() as batch:
        for i in range(n):
            ts = base + timedelta(weeks=i)
            occ = occupied_base + (i % 5)
            cap = 100
            batch.put_item(Item={
                "pk":              f"HOSPITAL#{hospital_id}",
                "sk":              f"SNAPSHOT#{ts.isoformat()}",
                "hospital_id":     hospital_id,
                "timestamp":       ts.isoformat(),
                "icu_capacity":    Decimal(str(cap)),
                "icu_occupied":    Decimal(str(occ)),
                "occupancy_ratio": Decimal(str(round(occ / cap, 4))),
            })
            rows.append({"timestamp": ts, "icu_occupied": float(occ), "icu_capacity": float(cap)})
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def _baseline(df):
    m = BaselineForecaster(target_col="icu_occupied")
    m.fit(df)
    return m, "baseline"


# ── batch ─────────────────────────────────────────────────────────────────────


@mock_aws
def test_batch_invocation_processes_multiple_hospitals():
    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    s3 = boto3.client("s3", region_name="us-east-1")
    _make_tables(ddb)
    s3.create_bucket(Bucket=DATA_BUCKET)
    df1 = _seed(ddb, "H001", occupied_base=60)
    df2 = _seed(ddb, "H002", occupied_base=80)

    with patch("lambdas.forecast.handler._select_model", return_value=_baseline(df1)):
        result = fc_handler.lambda_handler({"hospital_ids": ["H001", "H002"]}, _CTX)

    assert result["status"] == "complete"
    assert result["batched"] is True
    ids = {r["hospital_id"] for r in result["hospitals"]}
    assert ids == {"H001", "H002"}


# ── priority ─────────────────────────────────────────────────────────────────


@mock_aws
def test_priority_ordering_puts_high_occupancy_first():
    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    s3 = boto3.client("s3", region_name="us-east-1")
    _make_tables(ddb)
    s3.create_bucket(Bucket=DATA_BUCKET)
    _seed(ddb, "low",  occupied_base=40)   # ratio ~ 0.40
    _seed(ddb, "high", occupied_base=92)   # ratio ~ 0.92

    seen_order: list[str] = []

    def fake_select(hospital_id, df):
        seen_order.append(hospital_id)
        return _baseline(df)

    with patch("lambdas.forecast.handler._select_model", side_effect=fake_select):
        result = fc_handler.lambda_handler(
            {"hospital_ids": ["low", "high"], "priority": True}, _CTX
        )

    assert result["priority_used"] is True
    assert seen_order[0] == "high"


# ── fallback ─────────────────────────────────────────────────────────────────


@mock_aws
def test_fallback_to_baseline_when_selector_fails(monkeypatch):
    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    s3 = boto3.client("s3", region_name="us-east-1")
    _make_tables(ddb)
    s3.create_bucket(Bucket=DATA_BUCKET)
    _seed(ddb, "H001")

    def boom(self, df):
        raise RuntimeError("selector boom")

    # Force every BestModelSelector attempt to fail; fallback path should kick in.
    monkeypatch.setattr(
        "ml.model_selection.selector.BestModelSelector.select_best_model",
        boom,
    )

    result = fc_handler.lambda_handler({"hospital_id": "H001"}, _CTX)
    h = next(r for r in result["hospitals"] if r["hospital_id"] == "H001")
    assert h["status"] == "success"
    assert h["model"] == "baseline"
    assert h["fallback_used"] is True


# ── workload-tier metadata ───────────────────────────────────────────────────


@mock_aws
def test_response_includes_workload_tier_and_bottleneck():
    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    s3 = boto3.client("s3", region_name="us-east-1")
    _make_tables(ddb)
    s3.create_bucket(Bucket=DATA_BUCKET)
    df = _seed(ddb, "H001")

    with patch("lambdas.forecast.handler._select_model", return_value=_baseline(df)):
        result = fc_handler.lambda_handler({"hospital_id": "H001"}, _CTX)

    h = next(r for r in result["hospitals"] if r["hospital_id"] == "H001")
    assert h["workload_tier"] in {"small", "medium", "large"}
    assert h["bottleneck"] in {"cpu_bound", "io_bound", "comm_bound", "balanced"}
    assert h["wall_ms"] > 0
