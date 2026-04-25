"""Benchmark Baseline, SARIMA, Prophet on real HHS hospital data.

Paper Q5 evidence — screenshot the printed table for the report.

Usage:
    python scripts/benchmark_models.py
    python scripts/benchmark_models.py --hospitals 010001 010005 --horizon 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ml.config import DEFAULT_CONFIG
from ml.models.baseline import BaselineForecaster
from ml.models.prophet_model import ProphetForecaster
from ml.models.sarima_model import SarimaForecaster
from ml.utils.metrics import summarize_metrics


def _load_hospital(hospital_id: str, data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, dtype={"hospital_id": str}, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    hdf = df[df["hospital_id"] == hospital_id].sort_values("timestamp").reset_index(drop=True)
    if hdf.empty:
        raise ValueError(f"Hospital {hospital_id!r} not found in {data_path.name}")
    return hdf


def _run_model(factory, df: pd.DataFrame, horizon: int) -> dict:
    split    = max(len(df) - horizon, int(len(df) * 0.8))
    train_df = df.iloc[:split].copy()
    test_df  = df.iloc[split:].copy()

    model = factory()
    t0 = time.perf_counter()
    if isinstance(model, ProphetForecaster):
        model.fit(train_df, timestamp_col="timestamp")
    else:
        model.fit(train_df)
    train_time = round(time.perf_counter() - t0, 3)

    t1 = time.perf_counter()
    raw = model.predict(horizon)
    infer_time = round(time.perf_counter() - t1, 4)

    if isinstance(raw, pd.DataFrame):
        col   = "yhat" if "yhat" in raw.columns else raw.select_dtypes("number").columns[0]
        preds = raw[col].values
    elif isinstance(raw, list):
        preds = np.array(raw, dtype=float)
    else:
        preds = np.asarray(raw, dtype=float)

    actuals = test_df["icu_occupied"].values
    k       = min(len(preds), len(actuals), horizon)
    m       = summarize_metrics(actuals[:k], preds[:k])
    return {**m, "train_time_s": train_time, "infer_time_s": infer_time, "n_train": len(train_df)}


def benchmark(hospitals: list[str], data_path: Path, horizon: int = 4) -> pd.DataFrame:
    factories = {
        "baseline": lambda: BaselineForecaster(target_col="icu_occupied"),
        "sarima":   lambda: SarimaForecaster(target_col="icu_occupied"),
        "prophet":  lambda: ProphetForecaster(target_col="icu_occupied"),
    }
    rows = []
    for hid in hospitals:
        try:
            hdf = _load_hospital(hid, data_path)
        except ValueError as e:
            print(f"  SKIP {hid}: {e}")
            continue
        if len(hdf) <= horizon + 4:
            print(f"  SKIP {hid}: only {len(hdf)} rows (need >{horizon + 4})")
            continue

        for model_name, factory in factories.items():
            print(f"  {hid} / {model_name:<10}", end=" ", flush=True)
            try:
                stats = _run_model(factory, hdf, horizon)
                print(f"RMSE={stats['rmse']:.2f}  MAE={stats['mae']:.2f}  train={stats['train_time_s']}s")
                rows.append({"hospital_id": hid, "model": model_name, **stats})
            except Exception as exc:
                print(f"FAILED: {exc}")
                rows.append({"hospital_id": hid, "model": model_name, "error": str(exc)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ICU forecasting models")
    parser.add_argument("--hospitals", nargs="*",
                        help="Hospital IDs to benchmark (default: first 3 in CSV)")
    parser.add_argument("--horizon", type=int, default=4,
                        help="Forecast horizon in weeks (default: 4)")
    args = parser.parse_args()

    data_path = ROOT / "data" / "cleaned_hhs_ml_ready.csv"
    if not data_path.exists():
        print(f"[ERROR] {data_path} not found. Run data preprocessing first.")
        sys.exit(1)

    df_ids = pd.read_csv(data_path, usecols=["hospital_id"], dtype=str, low_memory=False)
    available = sorted(df_ids["hospital_id"].unique().tolist())
    hospitals  = args.hospitals or available[:3]

    print(f"Benchmarking {len(hospitals)} hospital(s), horizon={args.horizon} weeks\n")
    results = benchmark(hospitals, data_path, args.horizon)

    if results.empty:
        print("[ERROR] No results produced.")
        sys.exit(1)

    numeric_cols = ["rmse", "mae", "mape", "train_time_s"]
    valid = results.dropna(subset=["rmse"])
    if not valid.empty:
        summary = valid.groupby("model")[numeric_cols].mean().round(3)
        print("\n╔══ Average across hospitals ══╗")
        print(summary.to_string())

    print("\n╔══ Per-hospital results ══╗")
    display_cols = [c for c in ["hospital_id", "model", "rmse", "mae", "mape", "train_time_s"] if c in results.columns]
    print(results[display_cols].to_string(index=False))

    out = ROOT / "artifacts" / "benchmark_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
