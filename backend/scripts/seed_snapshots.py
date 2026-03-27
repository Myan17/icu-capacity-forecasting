from datetime import datetime, timedelta
import random

from app.db.database import SessionLocal
from app.models.tables import Snapshot


def main():
    db = SessionLocal()

    hospital_id = "HOSPITAL_A"
    base_time = datetime.utcnow() - timedelta(hours=48)
    capacity = 100
    occupied = 65

    for i in range(48):
        ts = base_time + timedelta(hours=i)

        admissions = random.randint(2, 8)
        discharges = random.randint(1, 6)
        transfers = random.randint(0, 3)
        ed_arrivals = random.randint(5, 18)
        staffing_level = round(random.uniform(0.75, 1.0), 2)

        occupied = max(
            0,
            min(
                capacity,
                occupied + admissions - discharges + random.randint(-2, 2),
            ),
        )

        row = Snapshot(
            hospital_id=hospital_id,
            timestamp=ts,
            icu_capacity=capacity,
            icu_occupied=occupied,
            admissions=admissions,
            discharges=discharges,
            transfers=transfers,
            ed_arrivals=ed_arrivals,
            staffing_level=staffing_level,
        )
        db.add(row)

    db.commit()
    db.close()
    print("Seeded 48 hourly snapshots.")


if __name__ == "__main__":
    main()