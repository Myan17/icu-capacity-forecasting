from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from app.db.database import Base


class Snapshot(Base):
    __tablename__ = "snapshots"

    id = Column(Integer, primary_key=True, index=True)
    hospital_id = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)

    icu_capacity = Column(Integer, nullable=False)
    icu_occupied = Column(Integer, nullable=False)
    admissions = Column(Integer, nullable=False)
    discharges = Column(Integer, nullable=False)
    transfers = Column(Integer, nullable=False)
    ed_arrivals = Column(Integer, nullable=False)
    staffing_level = Column(Float, nullable=False)


class Forecast(Base):
    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True, index=True)
    hospital_id = Column(String, index=True, nullable=False)
    forecast_time = Column(DateTime, index=True, nullable=False)
    predicted_icu_occupied = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    hospital_id = Column(String, index=True, nullable=False)
    created_at = Column(DateTime, index=True, nullable=False)
    risk_level = Column(String, nullable=False)
    message = Column(Text, nullable=False)