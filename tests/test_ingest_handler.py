"""Moto-mocked integration tests for lambdas/ingest/handler.lambda_handler.

Each test provisions real DynamoDB + S3 tables in moto, invokes lambda_handler
with a synthetic event, and asserts on both the response and the DynamoDB state.
"""
from __future__ import annotations

import csv
import io
import os

import boto3
import pytest
from moto import mock_aws

# ── environment setup must happen BEFORE the handler imports boto3 ────────────

SNAPSHOTS_TABLE = "test-snapshots"
FORECASTS_TABLE = "test-forecasts"
ALERTS_TABLE    = "test-alerts"
DATA_BUCKET     = "test-data-bucket"

os.environ.setdefault("SNAPSHOTS_TABLE", SNAPSHOTS_TABLE)
os.environ.setdefault("FORECASTS_TABLE", FORECASTS_TABLE)
os.environ.setdefault("ALERTS_TABLE",    ALERTS_TABLE)
os.environ.setdefault("DATA_BUCKET",     DATA_BUCKET)
os.environ.setdefault("POWERTOOLS_SERVICE_NAME", "test")
os.environ.setdefault("POWERTOOLS_METRICS_NAMESPACE", "TestNS")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")

from lambdas.ingest.handler import lambda_handler  # noqa: E402


class _FakeContext:
    function_name          = "test-ingest"
    memory_limit_in_mb     = 512
    invoked_function_arn   = "arn:aws:lambda:us-east-1:000000000000:function:test-ingest"
    aws_request_id         = "test-request-id"


_CTX = _FakeContext()


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_tables(dynamodb):
    for table_name, sk_type in [
        (SNAPSHOTS_TABLE, "S"),
        (FORECASTS_TABLE, "S"),
        (ALERTS_TABLE,    "S"),
    ]:
        dynamodb.create_table(
            TableName=table_name,
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


def _make_bucket(s3):
    s3.create_bucket(Bucket=DATA_BUCKET)


def _csv_bytes(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


_GOOD_ROWS = [
    {
        "hospital_id":   "H001",
        "timestamp":     "2024-01-07",
        "icu_capacity":  "100",
        "icu_occupied":  "80",
        "admissions":    "5",
        "discharges":    "4",
        "transfers":     "1",
        "ed_arrivals":   "10",
        "staffing_level":"0.90",
    },
    {
        "hospital_id":   "H002",
        "timestamp":     "2024-01-07",
        "icu_capacity":  "50",
        "icu_occupied":  "40",
        "admissions":    "3",
        "discharges":    "2",
        "transfers":     "0",
        "ed_arrivals":   "8",
        "staffing_level":"0.95",
    },
]


# ── test 1: successful ingestion returns status=success ───────────────────────

@mock_aws
def test_success_status():
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    s3       = boto3.client("s3",         region_name="us-east-1")
    _make_tables(dynamodb)
    _make_bucket(s3)

    s3.put_object(Bucket=DATA_BUCKET, Key="raw/data.csv", Body=_csv_bytes(_GOOD_ROWS))

    result = lambda_handler({"s3_key": "raw/data.csv"}, _CTX)

    assert result["status"] == "success"
    assert result["rows_written"] == 2


# ── test 2: occupancy_ratio is written to DynamoDB ───────────────────────────

@mock_aws
def test_occupancy_ratio_written():
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    s3       = boto3.client("s3",         region_name="us-east-1")
    _make_tables(dynamodb)
    _make_bucket(s3)

    s3.put_object(Bucket=DATA_BUCKET, Key="raw/data.csv", Body=_csv_bytes(_GOOD_ROWS[:1]))

    lambda_handler({"s3_key": "raw/data.csv"}, _CTX)

    table = dynamodb.Table(SNAPSHOTS_TABLE)
    resp  = table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key("pk").eq("HOSPITAL#H001")
    )
    item = resp["Items"][0]
    ratio = float(item["occupancy_ratio"])
    assert abs(ratio - 0.80) < 0.01


# ── test 3: validation failure writes ingestion_failed alert ──────────────────

@mock_aws
def test_validation_failure_writes_alert():
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    s3       = boto3.client("s3",         region_name="us-east-1")
    _make_tables(dynamodb)
    _make_bucket(s3)

    bad_rows = [{**_GOOD_ROWS[0], "icu_occupied": "-999"}]
    s3.put_object(Bucket=DATA_BUCKET, Key="raw/bad.csv", Body=_csv_bytes(bad_rows))

    result = lambda_handler({"s3_key": "raw/bad.csv", "hospital_id": "H001"}, _CTX)

    assert result["status"] == "validation_failed"

    alert_table = dynamodb.Table(ALERTS_TABLE)
    alerts = alert_table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key("pk").eq("HOSPITAL#H001")
    )
    assert len(alerts["Items"]) >= 1
    assert alerts["Items"][0]["risk_level"] == "ingestion_failed"


# ── test 4: hospital_id filter ingests only matching rows ─────────────────────

@mock_aws
def test_hospital_id_filter():
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    s3       = boto3.client("s3",         region_name="us-east-1")
    _make_tables(dynamodb)
    _make_bucket(s3)

    s3.put_object(Bucket=DATA_BUCKET, Key="raw/data.csv", Body=_csv_bytes(_GOOD_ROWS))

    result = lambda_handler({"s3_key": "raw/data.csv", "hospital_id": "H001"}, _CTX)

    assert result["status"] == "success"
    assert result["rows_written"] == 1

    table = dynamodb.Table(SNAPSHOTS_TABLE)
    h2_items = table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key("pk").eq("HOSPITAL#H002")
    )
    assert len(h2_items["Items"]) == 0


# ── test 5: empty S3 object returns no_data ───────────────────────────────────

@mock_aws
def test_empty_csv_returns_no_data():
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    s3       = boto3.client("s3",         region_name="us-east-1")
    _make_tables(dynamodb)
    _make_bucket(s3)

    s3.put_object(Bucket=DATA_BUCKET, Key="raw/empty.csv", Body=b"hospital_id,timestamp,icu_capacity,icu_occupied\n")

    result = lambda_handler({"s3_key": "raw/empty.csv"}, _CTX)

    assert result["status"] == "no_data"


# ── test 6: missing required column returns validation_failed ─────────────────

@mock_aws
def test_missing_required_column():
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    s3       = boto3.client("s3",         region_name="us-east-1")
    _make_tables(dynamodb)
    _make_bucket(s3)

    bad_rows = [{"hospital_id": "H001", "timestamp": "2024-01-07", "icu_occupied": "80"}]
    s3.put_object(Bucket=DATA_BUCKET, Key="raw/bad.csv", Body=_csv_bytes(bad_rows))

    result = lambda_handler({"s3_key": "raw/bad.csv", "hospital_id": "H001"}, _CTX)

    assert result["status"] == "validation_failed"
    assert "icu_capacity" in result["detail"]["summary"]
