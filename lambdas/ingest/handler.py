"""Ingest Lambda — Phase 2 core implementation.

Flow:
  1. Parse HHS snapshot CSV/JSON from S3 (or direct event payload)
  2. Run Great Expectations validation suite
  3. Write clean snapshot items to DynamoDB
  4. On validation failure: write ingestion_failed alert to DynamoDB
  5. Publish GE Data Docs HTML to S3 (data-docs/ prefix)

Observability: AWS Lambda Powertools structured logging + custom metrics.
"""
from __future__ import annotations

import io
import json
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import boto3
import pandas as pd
from aws_lambda_powertools import Logger, Metrics, Tracer
from aws_lambda_powertools.metrics import MetricUnit

# Shared helpers (copied into Lambda package at build time)
from shared.dynamo import (
    alert_sk,
    alerts_table,
    snapshot_pk,
    snapshot_sk,
    snapshots_table,
)
from shared.models import AlertItem, SnapshotItem
from lambdas.ingest.validation_suite import run_validation, run_ge_suite, _GE_AVAILABLE

logger  = Logger(service="hospital-ingest")
tracer  = Tracer(service="hospital-ingest")
metrics = Metrics(namespace="HospitalForecasting", service="hospital-ingest")

S3 = boto3.client("s3")
DATA_BUCKET = os.environ.get("DATA_BUCKET", "")

# Required HHS columns → internal field mapping
HHS_COLUMN_MAP = {
    "hospital_pk":                       "hospital_id",
    "collection_week":                   "timestamp",
    "inpatient_beds_used_7_day_avg":     "icu_occupied",
    "inpatient_beds_7_day_avg":         "icu_capacity",
    "total_adult_patients_hospitalized_confirmed_and_suspected_covid_7_day_avg": "ed_arrivals",
    "staffed_icu_adult_patients_confirmed_covid_7_day_avg": "staffing_level",
}

REQUIRED_COLUMNS = [
    "hospital_id", "timestamp", "icu_capacity", "icu_occupied",
]


# ── Entry point ───────────────────────────────────────────────────────────────

@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: Any) -> dict:
    hospital_id: str | None = event.get("hospital_id")
    s3_bucket: str = event.get("s3_bucket", DATA_BUCKET)
    s3_key: str | None = event.get("s3_key")

    if s3_key:
        df = _load_from_s3(s3_bucket, s3_key)
    else:
        df = _load_default_dataset(s3_bucket)

    if df is None or df.empty:
        logger.warning("No snapshot data found — nothing to ingest")
        return {"status": "no_data"}

    # Normalize column names from HHS schema if needed
    df = _normalize_columns(df)

    # Filter to specific hospital if requested
    if hospital_id and "hospital_id" in df.columns:
        df = df[df["hospital_id"].astype(str) == str(hospital_id)]

    # Run Great Expectations validation
    validation_result = _run_validation(df)

    if not validation_result["success"]:
        _write_ingestion_failed_alert(
            hospital_id=hospital_id or "UNKNOWN",
            reason=validation_result["summary"],
        )
        metrics.add_metric(name="IngestValidationFailures", unit=MetricUnit.Count, value=1)
        logger.error("Validation failed", extra=validation_result)
        return {"status": "validation_failed", "detail": validation_result}

    # Write valid rows to DynamoDB
    written = _write_snapshots(df)
    metrics.add_metric(name="SnapshotsIngested", unit=MetricUnit.Count, value=written)
    logger.info("Ingest complete", extra={"rows_written": written})

    return {"status": "success", "rows_written": written}


# ── S3 data loading ───────────────────────────────────────────────────────────

@tracer.capture_method
def _load_from_s3(bucket: str, key: str) -> pd.DataFrame | None:
    try:
        obj = S3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read()
        if key.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(body))
        return pd.read_csv(io.StringIO(body.decode("utf-8")))
    except Exception as exc:
        logger.error("Failed to load from S3", extra={"bucket": bucket, "key": key, "error": str(exc)})
        return None


def _load_default_dataset(bucket: str) -> pd.DataFrame | None:
    """Fallback: load the pre-processed HHS file from S3."""
    key = "raw/cleaned_hhs_ml_ready.csv"
    return _load_from_s3(bucket, key)


# ── Column normalisation ──────────────────────────────────────────────────────

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=HHS_COLUMN_MAP)
    if "hospital_id" not in df.columns and "hospital_pk" not in df.columns:
        # Try the 'group_col' convention used in ml/config.py
        if "hospital_id" not in df.columns:
            logger.warning("hospital_id column missing after normalization")
    # Coerce types
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ["icu_capacity", "icu_occupied", "admissions", "discharges", "transfers", "ed_arrivals"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    if "staffing_level" in df.columns:
        df["staffing_level"] = pd.to_numeric(df["staffing_level"], errors="coerce").fillna(0.85)
    return df


# ── Great Expectations validation ─────────────────────────────────────────────

def _run_validation(df: pd.DataFrame) -> dict:
    """Delegate to validation_suite.run_validation; publish Data Docs on completion."""
    result = run_validation(df)
    if _GE_AVAILABLE:
        _publish_data_docs(df, result["success"])
    return result


def _publish_data_docs(df: pd.DataFrame, success: bool) -> None:
    """Write a minimal Data Docs HTML summary to S3."""
    if not DATA_BUCKET:
        return
    rows = len(df)
    status_class = "pass" if success else "fail"
    status_label = "PASSED" if success else "FAILED"
    html = f"""<!DOCTYPE html>
<html>
<head><title>Hospital Forecasting — Data Docs</title>
<style>
  body {{ font-family: sans-serif; margin: 2em; }}
  .pass {{ color: green; }} .fail {{ color: red; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
  th {{ background: #f5f5f5; }}
</style></head>
<body>
<h1>Snapshot Validation Report</h1>
<p>Generated: {datetime.now(timezone.utc).isoformat()}</p>
<p>Rows checked: <strong>{rows}</strong></p>
<p>Status: <strong class="{status_class}">{status_label}</strong></p>
<h2>Column Stats</h2>
<table>
<tr><th>Column</th><th>Nulls</th><th>Min</th><th>Max</th><th>Mean</th></tr>
"""
    for col in ["icu_capacity", "icu_occupied", "staffing_level"]:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            html += (
                f"<tr><td>{col}</td>"
                f"<td>{series.isna().sum()}</td>"
                f"<td>{series.min():.1f}</td>"
                f"<td>{series.max():.1f}</td>"
                f"<td>{series.mean():.2f}</td></tr>\n"
            )
    html += "</table></body></html>"

    try:
        S3.put_object(
            Bucket=DATA_BUCKET,
            Key=f"data-docs/validation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.html",
            Body=html.encode("utf-8"),
            ContentType="text/html",
        )
        logger.info("Data Docs published to S3")
    except Exception as exc:
        logger.warning("Could not publish Data Docs", extra={"error": str(exc)})


# ── DynamoDB writes ───────────────────────────────────────────────────────────

@tracer.capture_method
def _write_snapshots(df: pd.DataFrame) -> int:
    table = snapshots_table()
    written = 0
    with table.batch_writer() as batch:
        for _, row in df.iterrows():
            ts: datetime = row["timestamp"]
            if pd.isna(ts):
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            hospital_id = str(row.get("hospital_id", "UNKNOWN"))
            icu_capacity = int(row.get("icu_capacity", 0))
            icu_occupied  = int(row.get("icu_occupied", 0))
            item = {
                "pk":              snapshot_pk(hospital_id),
                "sk":              snapshot_sk(ts),
                "hospital_id":     hospital_id,
                "timestamp":       ts.isoformat(),
                "icu_capacity":    icu_capacity,
                "icu_occupied":    icu_occupied,
                "admissions":      int(row.get("admissions", 0)),
                "discharges":      int(row.get("discharges", 0)),
                "transfers":       int(row.get("transfers", 0)),
                "ed_arrivals":     int(row.get("ed_arrivals", 0)),
                "staffing_level":  Decimal(str(round(float(row.get("staffing_level", 0.85)), 4))),
                "occupancy_ratio": Decimal(str(round(icu_occupied / max(icu_capacity, 1), 4))),
            }
            batch.put_item(Item=item)
            written += 1
    return written


def _write_ingestion_failed_alert(hospital_id: str, reason: str) -> None:
    now = datetime.now(timezone.utc)
    table = alerts_table()
    table.put_item(Item={
        "pk":           snapshot_pk(hospital_id),
        "sk":           alert_sk(now),
        "hospital_id":  hospital_id,
        "created_at":   now.isoformat(),
        "risk_level":   "ingestion_failed",
        "message":      f"Ingestion validation failed: {reason}",
        "breach_prob":  Decimal("0"),
        "dedupe_key":   "",
    })
    logger.warning("ingestion_failed alert written", extra={"hospital_id": hospital_id, "reason": reason})
