"""DynamoDB key builders and thin client helpers.

Key design (matches template.yaml table definitions):
  snapshots  : PK = HOSPITAL#{id}   SK = SNAPSHOT#{iso}
  forecasts  : PK = HOSPITAL#{id}   SK = FORECAST#{run_id}#{step_iso}
  alerts     : PK = HOSPITAL#{id}   SK = ALERT#{iso}
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key

_resource = None


def _get_resource():
    global _resource
    if _resource is None:
        _resource = boto3.resource("dynamodb")
    return _resource


def _table(name_env_var: str):
    table_name = os.environ[name_env_var]
    return _get_resource().Table(table_name)


def snapshots_table():
    return _table("SNAPSHOTS_TABLE")


def forecasts_table():
    return _table("FORECASTS_TABLE")


def alerts_table():
    return _table("ALERTS_TABLE")


# ── Key builders ──────────────────────────────────────────────────────────────

def snapshot_pk(hospital_id: str) -> str:
    return f"HOSPITAL#{hospital_id}"


def snapshot_sk(ts: datetime) -> str:
    return f"SNAPSHOT#{ts.strftime('%Y-%m-%dT%H:%M:%SZ')}"


def forecast_sk(run_id: str, step_ts: datetime) -> str:
    return f"FORECAST#{run_id}#{step_ts.strftime('%Y-%m-%dT%H:%M:%SZ')}"


def alert_sk(ts: datetime) -> str:
    return f"ALERT#{ts.strftime('%Y-%m-%dT%H:%M:%SZ')}"


def alert_dedupe_key(hospital_id: str, risk_level: str, breach_window_ts: datetime) -> str:
    """Stable hash used to deduplicate repeated alerts for the same breach window."""
    raw = f"{hospital_id}#{risk_level}#{breach_window_ts.strftime('%Y-%W')}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Query helpers ─────────────────────────────────────────────────────────────

def query_recent_snapshots(hospital_id: str, limit: int = 52) -> list[dict[str, Any]]:
    """Return the most recent `limit` snapshots for a hospital (newest last)."""
    table = snapshots_table()
    resp = table.query(
        KeyConditionExpression=Key("pk").eq(snapshot_pk(hospital_id)),
        ScanIndexForward=False,
        Limit=limit,
    )
    items = resp.get("Items", [])
    return list(reversed(items))  # chronological order


def query_latest_snapshot(hospital_id: str) -> dict[str, Any] | None:
    items = query_recent_snapshots(hospital_id, limit=1)
    return items[0] if items else None


def query_forecasts(hospital_id: str, run_id: str | None = None) -> list[dict[str, Any]]:
    """Return all forecast steps for a hospital, optionally filtered by run_id."""
    table = forecasts_table()
    pk = snapshot_pk(hospital_id)
    if run_id:
        resp = table.query(
            KeyConditionExpression=(
                Key("pk").eq(pk) & Key("sk").begins_with(f"FORECAST#{run_id}#")
            ),
            ScanIndexForward=True,
        )
    else:
        resp = table.query(
            KeyConditionExpression=Key("pk").eq(pk),
            ScanIndexForward=True,
        )
    return resp.get("Items", [])


def query_alerts(hospital_id: str, limit: int = 50) -> list[dict[str, Any]]:
    table = alerts_table()
    resp = table.query(
        KeyConditionExpression=Key("pk").eq(snapshot_pk(hospital_id)),
        ScanIndexForward=False,
        Limit=limit,
    )
    return resp.get("Items", [])


def alert_exists(hospital_id: str, dedupe_key: str) -> bool:
    """Check if an alert with this dedupe_key already exists (prevents spam)."""
    table = alerts_table()
    resp = table.query(
        KeyConditionExpression=Key("pk").eq(snapshot_pk(hospital_id)),
        FilterExpression=Key("dedupe_key").eq(dedupe_key)
        if hasattr(Key("dedupe_key"), "eq")
        else None,
        ScanIndexForward=False,
        Limit=20,
    )
    for item in resp.get("Items", []):
        if item.get("dedupe_key") == dedupe_key:
            return True
    return False
