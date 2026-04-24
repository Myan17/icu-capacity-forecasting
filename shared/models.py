"""Pydantic models shared across Lambda functions and backend.

All DynamoDB items are represented here so the schema is defined once.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Snapshot ─────────────────────────────────────────────────────────────────

class SnapshotItem(BaseModel):
    """One weekly ICU occupancy snapshot for a hospital."""
    hospital_id:    str
    timestamp:      datetime
    icu_capacity:   int
    icu_occupied:   int
    admissions:     int
    discharges:     int
    transfers:      int
    ed_arrivals:    int
    staffing_level: float

    # Derived fields (computed on ingest)
    occupancy_ratio: float = Field(default=0.0)

    def model_post_init(self, _context) -> None:
        if self.icu_capacity > 0:
            self.occupancy_ratio = round(self.icu_occupied / self.icu_capacity, 4)


# ── Forecast ─────────────────────────────────────────────────────────────────

class ForecastItem(BaseModel):
    """Single forecast step with confidence interval."""
    hospital_id:           str
    run_id:                str        # UUID generated per forecast run
    forecast_time:         datetime   # wall-clock time this step represents
    predicted_icu_occupied: float
    yhat_lower:            float      # lower confidence bound (95%)
    yhat_upper:            float      # upper confidence bound (95%)
    risk_level:            str        # GREEN | YELLOW | RED
    model_name:            str        # baseline | sarima | prophet
    breach_prob:           float = 0.0  # P(occupancy > RED threshold)


# ── Alert ─────────────────────────────────────────────────────────────────────

class AlertItem(BaseModel):
    """Risk alert — created when breach_prob exceeds threshold or point estimate crosses threshold."""
    hospital_id:      str
    created_at:       datetime
    risk_level:       str            # YELLOW | RED | ingestion_failed
    message:          str
    breach_prob:      float = 0.0    # 0.0 for point-threshold alerts
    breach_eta_hours: Optional[float] = None  # hours until projected breach
    dedupe_key:       str = ""       # hash(hospital_id + breach_window) prevents spam


# ── Ingestion event ───────────────────────────────────────────────────────────

class IngestEvent(BaseModel):
    """EventBridge / direct invocation payload for the ingest Lambda."""
    s3_bucket: Optional[str] = None
    s3_key:    Optional[str] = None
    hospital_id: Optional[str] = None  # if None, process all hospitals in the file


class ForecastEvent(BaseModel):
    """EventBridge / direct invocation payload for the forecast Lambda."""
    hospital_id: Optional[str] = None  # if None, forecast all hospitals in DynamoDB
