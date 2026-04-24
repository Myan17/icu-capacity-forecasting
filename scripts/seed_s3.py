"""Upload HHS snapshot CSV to S3 so the Ingest Lambda can consume it.

Usage:
    python scripts/seed_s3.py --bucket <bucket-name> [--key raw/cleaned_hhs_ml_ready.csv]
    python scripts/seed_s3.py --bucket <bucket-name> --file path/to/custom.csv

The Ingest Lambda looks for: s3://<bucket>/raw/cleaned_hhs_ml_ready.csv by default.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3

ROOT = Path(__file__).parent.parent
DEFAULT_CSV = ROOT / "data" / "cleaned_hhs_ml_ready.csv"
DEFAULT_KEY = "raw/cleaned_hhs_ml_ready.csv"


def seed(bucket: str, key: str, local_file: Path) -> None:
    if not local_file.exists():
        print(f"[ERROR] {local_file} not found. Run the ML preprocessing pipeline first.")
        sys.exit(1)

    s3 = boto3.client("s3")
    size_mb = local_file.stat().st_size / 1_048_576
    print(f"Uploading {local_file.name} ({size_mb:.1f} MB) → s3://{bucket}/{key}")

    s3.upload_file(
        Filename=str(local_file),
        Bucket=bucket,
        Key=key,
        ExtraArgs={"ContentType": "text/csv"},
    )
    print(f"[OK] s3://{bucket}/{key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed S3 with HHS snapshot CSV for Lambda ingest")
    parser.add_argument("--bucket", required=True, help="S3 bucket name (DATA_BUCKET)")
    parser.add_argument("--key",    default=DEFAULT_KEY, help="S3 key to write to")
    parser.add_argument("--file",   default=str(DEFAULT_CSV), help="Local CSV file path")
    args = parser.parse_args()

    seed(bucket=args.bucket, key=args.key, local_file=Path(args.file))
