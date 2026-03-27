from datetime import datetime, timedelta

import pandas as pd
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.tables import Alert, Forecast, Snapshot
from app.schemas.snapshot import SnapshotCreate
from app.services.forecast import naive_forecast_next_24h
from app.services.risk import compute_risk

router = APIRouter()


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
def run_forecast(hospital_id: str, db: Session = Depends(get_db)):
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

    preds = naive_forecast_next_24h(df)
    latest_capacity = int(df["icu_capacity"].iloc[-1])
    start_time = datetime.utcnow()

    created = []
    for i, pred in enumerate(preds):
        forecast_time = start_time + timedelta(hours=i + 1)
        risk_level = compute_risk(pred, latest_capacity)

        forecast_row = Forecast(
            hospital_id=hospital_id,
            forecast_time=forecast_time,
            predicted_icu_occupied=pred,
            risk_level=risk_level,
        )
        db.add(forecast_row)
        created.append(
            {
                "forecast_time": forecast_time.isoformat(),
                "predicted_icu_occupied": pred,
                "risk_level": risk_level,
            }
        )

        if risk_level in {"YELLOW", "RED"}:
            alert = Alert(
                hospital_id=hospital_id,
                created_at=datetime.utcnow(),
                risk_level=risk_level,
                message=f"Predicted ICU occupancy risk {risk_level} at {forecast_time.isoformat()}",
            )
            db.add(alert)

    db.commit()
    return {"hospital_id": hospital_id, "forecasts": created}


@router.get("/forecasts/{hospital_id}")
def get_forecasts(hospital_id: str, db: Session = Depends(get_db)):
    rows = (
        db.query(Forecast)
        .filter(Forecast.hospital_id == hospital_id)
        .order_by(Forecast.forecast_time.asc())
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