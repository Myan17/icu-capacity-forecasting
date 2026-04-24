"""Upload trained model artifacts from ml/artifacts/models/ to S3.

Run after training to make artifacts available to the Forecast Lambda.

Usage:
    python scripts/upload_artifacts_to_s3.py --bucket <bucket-name> [--env dev]

The Forecast Lambda looks for: s3://<bucket>/artifacts/models/<hospital_id>_best.joblib
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import boto3
import joblib

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def upload(bucket: str) -> None:
    s3 = boto3.client("s3")
    model_dir = ROOT / "ml" / "artifacts" / "models"

    if not model_dir.exists():
        print(f"[ERROR] {model_dir} does not exist. Run ml/train.py first.")
        sys.exit(1)

    metadata_files = list(model_dir.glob("*.metadata.json"))
    if not metadata_files:
        print("[ERROR] No metadata files found. Train models first.")
        sys.exit(1)

    # Select the best (lowest RMSE) artifact per hospital
    best: dict[str, dict] = {}
    for meta_path in metadata_files:
        with open(meta_path) as f:
            meta = json.load(f)
        hid  = meta.get("hospital_id", "")
        rmse = meta.get("metrics", {}).get("rmse", float("inf"))
        pkl  = meta_path.with_suffix("").with_suffix(".pkl")
        if pkl.exists() and (hid not in best or rmse < best[hid]["rmse"]):
            best[hid] = {"rmse": rmse, "pkl": pkl, "model_name": meta.get("model_name", "unknown")}

    if not best:
        print("[ERROR] No .pkl files matched metadata. Train first.")
        sys.exit(1)

    uploaded = 0
    for hid, info in best.items():
        # joblib.load handles .pkl files (same underlying format, safer API)
        model_obj = joblib.load(str(info["pkl"]))
        bundle = {"model": model_obj, "model_name": info["model_name"]}

        joblib_path = info["pkl"].with_suffix(".joblib")
        joblib.dump(bundle, joblib_path)

        s3_key = f"artifacts/models/{hid}_best.joblib"
        s3.upload_file(str(joblib_path), bucket, s3_key)
        print(f"  {hid} ({info['model_name']}, RMSE={info['rmse']:.3f}) → s3://{bucket}/{s3_key}")
        uploaded += 1

    print(f"\n[OK] {uploaded} artifact(s) uploaded to s3://{bucket}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()
    upload(args.bucket)
