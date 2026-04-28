"""Parallel model selector — Lecture 3 (Distributed ML / Alpa) data parallelism.

The original ``BestModelSelector`` fits Baseline → Prophet → SARIMA *sequentially*
on the same train split. Each candidate is independent (no shared state, no
gradients to synchronise), so this is embarrassingly parallel — exactly the
pattern data-parallel training exploits at scale.

This module provides a drop-in replacement that fans out the candidate fits
onto a thread or process pool. We default to threads because:
  * Prophet, statsmodels, and our Baseline release the GIL during the heavy
    BLAS / Stan calls; threads give near-linear speed-up without pickling cost.
  * Forecast Lambdas have ≤ 2 vCPUs in practice; spawning processes hurts more
    than it helps.

A process executor is exposed for benchmarks (``executor="process"``).

Communication overhead is recorded in a ``WorkloadProfiler`` so the
auto-optimizer can decide whether parallelism is worth it for a given hospital.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from ml.model_selection.selector import BestModelSelector, ModelSelectionResult
from shared.workload_metrics import WorkloadKind, WorkloadProfiler


@dataclass
class ParallelSelectionResult:
    inner: ModelSelectionResult
    executor: str
    num_workers: int
    fit_durations_ms: Dict[str, float]
    total_wall_ms: float
    sequential_estimate_ms: float

    @property
    def speedup(self) -> float:
        if self.total_wall_ms <= 0:
            return 1.0
        return self.sequential_estimate_ms / self.total_wall_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_model_name":      self.inner.best_model_name,
            "best_score":           self.inner.best_score,
            "metric_used":          self.inner.metric_used,
            "executor":             self.executor,
            "num_workers":          self.num_workers,
            "fit_durations_ms":     self.fit_durations_ms,
            "total_wall_ms":        round(self.total_wall_ms, 2),
            "sequential_estimate_ms": round(self.sequential_estimate_ms, 2),
            "speedup":              round(self.speedup, 2),
        }


class ParallelBestModelSelector(BestModelSelector):
    """Same API as BestModelSelector but fits candidates concurrently."""

    def __init__(
        self,
        executor: Literal["thread", "process"] = "thread",
        num_workers: Optional[int] = None,
        profiler: Optional[WorkloadProfiler] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.executor = executor
        self.num_workers = num_workers or min(len(self.candidate_models), os.cpu_count() or 2)
        self.profiler = profiler

    def select_best_model_parallel(self, df: pd.DataFrame) -> ParallelSelectionResult:
        df = df.sort_values(self.timestamp_col).reset_index(drop=True)

        if len(df) <= self.horizon:
            raise ValueError(
                f"Not enough rows to do model selection. "
                f"Need more than horizon={self.horizon}, got {len(df)} rows."
            )

        train_df = df.iloc[:-self.horizon].copy()
        valid_df = df.iloc[-self.horizon:].copy()

        actual_values = valid_df[self.target_col].to_numpy()

        Pool: type
        if self.executor == "process":
            Pool = ProcessPoolExecutor
        else:
            Pool = ThreadPoolExecutor

        fit_durations: Dict[str, float] = {}
        all_results: List[Dict[str, Any]] = []

        wall_start = time.perf_counter()
        with Pool(max_workers=self.num_workers) as pool:
            futures: Dict[Future, str] = {}
            for name, factory in self.candidate_models.items():
                fut = pool.submit(_fit_and_score, factory, train_df, self.horizon,
                                  self.timestamp_col, self.target_col, name)
                futures[fut] = name

            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    all_results.append({"model_name": name, "status": "failed", "error": str(exc)})
                    fit_durations[name] = -1.0
                    continue
                fit_durations[name] = res["fit_ms"]
                all_results.append({
                    "model_name":  name,
                    "status":      "success",
                    "predictions": res["predictions"].tolist(),
                    "actuals":     actual_values[:len(res["predictions"])].tolist(),
                    **res["metrics"],
                })

        total_wall_ms = (time.perf_counter() - wall_start) * 1000.0

        successes = [r for r in all_results if r.get("status") == "success"]
        if not successes:
            raise RuntimeError("All candidate models failed during parallel selection.")

        best = min(successes, key=lambda r: r[self.metric])
        best_name = best["model_name"]

        if self.profiler:
            with self.profiler.stage(f"refit_full_{best_name}", WorkloadKind.CPU):
                final_model = self.candidate_models[best_name]()
                self._fit_model(final_model, df)
        else:
            final_model = self.candidate_models[best_name]()
            self._fit_model(final_model, df)

        inner = ModelSelectionResult(
            best_model_name=best_name,
            best_model=final_model,
            best_score=best[self.metric],
            metric_used=self.metric,
            all_results=all_results,
            train_rows=len(train_df),
            validation_rows=len(valid_df),
        )

        sequential_estimate_ms = sum(d for d in fit_durations.values() if d >= 0)

        return ParallelSelectionResult(
            inner=inner,
            executor=self.executor,
            num_workers=self.num_workers,
            fit_durations_ms=fit_durations,
            total_wall_ms=total_wall_ms,
            sequential_estimate_ms=sequential_estimate_ms,
        )


def _fit_and_score(
    factory: Callable[[], Any],
    train_df: pd.DataFrame,
    horizon: int,
    timestamp_col: str,
    target_col: str,
    model_name: str,
) -> Dict[str, Any]:
    """Worker function — runs in a separate thread/process.

    Kept module-level so it pickles cleanly for ProcessPoolExecutor."""
    from ml.models.prophet_model import ProphetForecaster

    t0 = time.perf_counter()
    model = factory()
    if isinstance(model, ProphetForecaster):
        model.fit(train_df, timestamp_col=timestamp_col)
    else:
        model.fit(train_df)
    preds = model.predict(horizon)

    if isinstance(preds, pd.DataFrame):
        if "yhat" in preds.columns:
            arr = preds["yhat"].to_numpy()
        else:
            arr = preds.iloc[:, 0].to_numpy()
    elif isinstance(preds, pd.Series):
        arr = preds.to_numpy()
    else:
        arr = np.asarray(preds, dtype=float)

    actual = train_df[target_col].to_numpy()
    fit_ms = (time.perf_counter() - t0) * 1000.0

    error = (actual[-len(arr):] if len(arr) <= len(actual) else actual) - arr[:min(len(arr), len(actual))]
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mape = (
        float(np.mean(np.abs(error / np.maximum(actual[-len(arr):], 1e-9))) * 100)
        if len(actual) >= len(arr)
        else float("inf")
    )

    return {
        "fit_ms":      fit_ms,
        "predictions": arr,
        "metrics":     {"mae": mae, "rmse": rmse, "mape": mape},
        "model_name":  model_name,
    }


__all__ = ["ParallelBestModelSelector", "ParallelSelectionResult"]
