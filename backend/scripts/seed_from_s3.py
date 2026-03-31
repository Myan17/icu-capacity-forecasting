from io import StringIO

import boto3
import pandas as pd

from app.db.database import SessionLocal
from app.models.tables import Snapshot

BUCKET = "icu-capacity-forecasting"
KEY = "cleaned_hhs_ml_ready.csv"


def safe_int(value, default=0):
    if pd.isna(value):
        return default
    return int(round(float(value)))


def main():
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET, Key=KEY)
    content = obj["Body"].read().decode("utf-8")

    df = pd.read_csv(StringIO(content), dtype={"hospital_id": str}, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # keep only columns we need
    df = df[["hospital_id", "timestamp", "icu_capacity", "icu_occupied"]].copy()

    # remove bad rows
    df = df.dropna(subset=["hospital_id", "timestamp", "icu_capacity", "icu_occupied"])

    db = SessionLocal()
    try:
        # optional: clear old snapshot rows before re-seeding
        deleted = db.query(Snapshot).delete()
        db.commit()
        print(f"Deleted {deleted} old snapshot rows")

        inserted = 0

        for _, row in df.iterrows():
            snapshot = Snapshot(
                hospital_id=str(row["hospital_id"]),
                timestamp=row["timestamp"].to_pydatetime(),
                icu_capacity=safe_int(row["icu_capacity"]),
                icu_occupied=safe_int(row["icu_occupied"]),
                admissions=0,
                discharges=0,
                transfers=0,
                ed_arrivals=0,
                staffing_level=0.85,
            )
            db.add(snapshot)
            inserted += 1

            if inserted % 1000 == 0:
                db.commit()
                print(f"Inserted {inserted} rows...")

        db.commit()
        print(f"Finished. Inserted {inserted} snapshot rows.")

    finally:
        db.close()


if __name__ == "__main__":
    main()
