from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from app.db.database import Base


class Snapshot(Base):
    __tablename__ = "snapshots"

    id              = Column(Integer, primary_key=True, index=True)
    hospital_id     = Column(String, index=True, nullable=False)
    timestamp       = Column(DateTime, index=True, nullable=False)
    icu_capacity    = Column(Integer, nullable=False)
    icu_occupied    = Column(Integer, nullable=False)
    admissions      = Column(Integer, nullable=False)
    discharges      = Column(Integer, nullable=False)
    transfers       = Column(Integer, nullable=False)
    ed_arrivals     = Column(Integer, nullable=False)
    staffing_level  = Column(Float, nullable=False)
    occupancy_ratio = Column(Float, nullable=True)   # icu_occupied / icu_capacity


class Forecast(Base):
    __tablename__ = "forecasts"

    id                      = Column(Integer, primary_key=True, index=True)
    hospital_id             = Column(String, index=True, nullable=False)
    forecast_time           = Column(DateTime, index=True, nullable=False)
    predicted_icu_occupied  = Column(Float, nullable=False)
    # Confidence interval (95%) — populated when ML model provides intervals
    yhat_lower              = Column(Float, nullable=True)
    yhat_upper              = Column(Float, nullable=True)
    risk_level              = Column(String, nullable=False)
    model_name              = Column(String, nullable=True)   # baseline | sarima | prophet
    # Probabilistic alerting fields
    breach_prob             = Column(Float, nullable=True, default=0.0)


class Alert(Base):
    __tablename__ = "alerts"

    id               = Column(Integer, primary_key=True, index=True)
    hospital_id      = Column(String, index=True, nullable=False)
    created_at       = Column(DateTime, index=True, nullable=False)
    risk_level       = Column(String, nullable=False)
    message          = Column(Text, nullable=False)
    breach_prob      = Column(Float, nullable=True, default=0.0)
    breach_eta_hours = Column(Float, nullable=True)
    dedupe_key       = Column(String, nullable=True, index=True)