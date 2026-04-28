"""Benchmark every parallelisation strategy and pick the fastest acceptable one.

This is the ergonomic CLI wrapper around ``ml.pipelines.auto_optimizer.AutoPipelineOptimizer``
(Lecture 3 / Alpa-style auto pipeline optimization).

Usage:
    python -m scripts.benchmark_pipelines --hospital hosp_000 --rows 120
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ml.pipelines.auto_optimizer import ALL_STRATEGIES, AutoPipelineOptimizer


def _synthetic_series(rows: int, seed: int = 0, capacity: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2023-01-01", periods=rows, freq="W")
    occ = (
        capacity * 0.6
        + 8 * np.sin(np.arange(rows) * 0.25)
        + rng.normal(0, 6, rows)
    )
    return pd.DataFrame({
        "timestamp":    timestamps,
        "icu_occupied": np.clip(occ, 1, capacity),
        "icu_capacity": capacity,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hospital", default="hosp_000")
    ap.add_argument("--rows",     type=int, default=120)
    ap.add_argument("--horizon",  type=int, default=4)
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--strategies", nargs="*", choices=list(ALL_STRATEGIES),
                    default=list(ALL_STRATEGIES))
    ap.add_argument("--rmse-tolerance", type=float, default=1.20,
                    help="Acceptable RMSE multiplier above the best (default 1.20 = 20%)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    df = _synthetic_series(args.rows, seed=args.seed)
    optimizer = AutoPipelineOptimizer(
        horizon=args.horizon,
        strategies=tuple(args.strategies),
        rmse_tolerance=args.rmse_tolerance,
    )
    decision = optimizer.explore(df, cache_key=args.hospital)

    if args.json:
        print(json.dumps(decision.to_dict(), indent=2))
        return

    print(f"=== Auto-optimizer benchmark for {args.hospital} ({args.rows} rows, horizon={args.horizon}) ===\n")
    print(f"{'strategy':<16} {'wall_ms':>10} {'rmse':>10} {'best_model':<12} status")
    print("-" * 60)
    for t in decision.trials:
        rmse_str = f"{t.rmse:.3f}" if np.isfinite(t.rmse) else "fail"
        status = "OK" if t.is_success() else f"FAIL: {t.error}"
        print(f"{t.strategy:<16} {t.wall_ms:>10.2f} {rmse_str:>10} {t.best_model:<12} {status}")
    print()
    print(f"Quality floor RMSE  : {decision.quality_floor_rmse:.3f}")
    print(f"Chosen strategy     : {decision.chosen}")
    print(f"  wall_ms          : {decision.chosen_wall_ms:.2f}")
    print(f"  rmse             : {decision.chosen_rmse:.3f}")


if __name__ == "__main__":
    main()
