"""Tests for lambdas/api/handler.py — focuses on the async forecast enqueue path."""
from __future__ import annotations

import json
import os
from unittest.mock import patch

import sys
from unittest.mock import MagicMock

# mangum is a Lambda-only package — stub it out before the handler module loads
sys.modules.setdefault("mangum", MagicMock())

# Powertools env must be set before the module is imported
os.environ.setdefault("POWERTOOLS_SERVICE_NAME",      "test")
os.environ.setdefault("POWERTOOLS_METRICS_NAMESPACE", "TestNS")
os.environ.setdefault("POWERTOOLS_TRACE_DISABLED",    "true")
os.environ.setdefault("AWS_DEFAULT_REGION",           "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID",            "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY",        "testing")
os.environ.setdefault("SNAPSHOTS_TABLE",              "test-snapshots")
os.environ.setdefault("FORECASTS_TABLE",              "test-forecasts")
os.environ.setdefault("ALERTS_TABLE",                 "test-alerts")
os.environ.setdefault("FORECAST_QUEUE_URL",           "https://sqs.us-east-1.amazonaws.com/123456789/test-queue")

from fastapi.testclient import TestClient          # noqa: E402
from lambdas.api.handler import app               # noqa: E402

client = TestClient(app, raise_server_exceptions=False)


# ── POST /forecast/{hospital_id} ──────────────────────────────────────────────

def test_trigger_forecast_returns_202():
    mock_sqs = MagicMock()
    with patch("lambdas.api.handler.boto3") as mock_boto3:
        mock_boto3.client.return_value = mock_sqs
        resp = client.post("/forecast/H001")

    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert body["hospital_id"] == "H001"


def test_trigger_forecast_sends_correct_sqs_message():
    mock_sqs = MagicMock()
    queue_url = os.environ["FORECAST_QUEUE_URL"]

    with patch("lambdas.api.handler.boto3") as mock_boto3:
        mock_boto3.client.return_value = mock_sqs
        client.post("/forecast/H002")

    mock_sqs.send_message.assert_called_once_with(
        QueueUrl=queue_url,
        MessageBody=json.dumps({"hospital_id": "H002"}),
    )


def test_trigger_forecast_sqs_error_returns_500():
    mock_sqs = MagicMock()
    mock_sqs.send_message.side_effect = Exception("SQS unavailable")

    with patch("lambdas.api.handler.boto3") as mock_boto3:
        mock_boto3.client.return_value = mock_sqs
        resp = client.post("/forecast/H003")

    assert resp.status_code == 500
