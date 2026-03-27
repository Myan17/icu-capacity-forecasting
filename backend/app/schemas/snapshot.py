from datetime import datetime
from pydantic import BaseModel


class SnapshotCreate(BaseModel):
    hospital_id: str
    timestamp: datetime
    icu_capacity: int
    icu_occupied: int
    admissions: int
    discharges: int
    transfers: int
    ed_arrivals: int
    staffing_level: float


class SnapshotOut(SnapshotCreate):
    id: int

    class Config:
        from_attributes = True