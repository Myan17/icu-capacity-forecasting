"""Alpa-style auto pipeline optimizer (Lecture 3).

Alpa's contribution is searching the *parallelism strategy space* and picking
the fastest one for a given model + workload. We reproduce that idea at a
much smaller scale: given a hospital's time series, benchmark each available
forecasting strategy on a held-out tail and pick the configuration with the
lowest wall-clock subject to a quality floor.

Strategies considered:
  * `sequential`     — original BestModelSelector, no parallelism (baseline)
  * `data_thread`    — ParallelBestModelSelector with thread pool
  * `data_process`   — ParallelBestModelSelector with process pool
  * `staged_inline`  — StagedPipeline with intra-op disabled (fewer threads)
  * `staged_intra`   — StagedPipeline with intra-op enabled (default)

The selector returns a ``StrategyDecision`` with timings, RMSEs, and a
recommendation that the forecast handler can record so future runs skip
re-exploration. Decisions are cached in-process; persisting them to S3 or
DynamoDB is left as an extension.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal

import numpy as np
import pandas as pd

from ml.model_selection.parallel_selector import ParallelBestModelSelector
from ml.model_selection.selector import BestModelSelector
from ml.pipelines.staged_pipeline import StagedPipeline


Strategy = Literal[
    "sequential",
    "data_thread",
    "data_process",
    "staged_inline",
    "staged_intra",
]

ALL_STRATEGIES: tuple[Strategy, ...] = (
    "sequential",
    "data_thread",
    "data_process",
    "staged_inline",
    "staged_intra",
)


@dataclass
class StrategyTrial:
    strategy: Strategy
    wall_ms: float
    best_model: str
    rmse: float
    error: str | None = None

    def is_success(self) -> bool:
        return self.error is None


@dataclass
class StrategyDecision:
    chosen: Strategy
    chosen_wall_ms: float
    chosen_rmse: float
    quality_floor_rmse: float
    trials: List[StrategyTrial] = field(default_factory=list)
    explored_strategies: List[Strategy] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chosen":             self.chosen,
            "chosen_wall_ms":     round(self.chosen_wall_ms, 2),
            "chosen_rmse":        round(self.chosen_rmse, 4),
            "quality_floor_rmse": round(self.quality_floor_rmse, 4),
            "trials": [
                {
                    "strategy":  t.strategy,
                    "wall_ms":   round(t.wall_ms, 2),
                    "rmse":      round(t.rmse, 4) if np.isfinite(t.rmse) else None,
                    "best_model": t.best_model,
                    "error":     t.error,
                }
                for t in self.trials
            ],
        }


_DECISION_CACHE: Dict[str, StrategyDecision] = {}


def _rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    n = min(len(actual), len(pred))
    if n == 0:
        return float("inf")
    return float(np.sqrt(np.mean((actual[:n] - pred[:n]) ** 2)))


def _trial_runners(horizon: int, timestamp_col: str, target_col: str, frequency: str):
    def run_sequential(df: pd.DataFrame) -> tuple[str, np.ndarray]:
        sel = BestModelSelector(horizon=horizon, timestamp_col=timestamp_col, target_col=target_col, frequency=frequency)
        res = sel.select_best_model(df)
        preds = np.asarray(res.best_model.predict(horizon), dtype=float)
        return res.best_model_name, preds

    def run_data(executor: str):
        def runner(df: pd.DataFrame) -> tuple[str, np.ndarray]:
            sel = ParallelBestModelSelector(executor=executor, horizon=horizon,
                                             timestamp_col=timestamp_col, target_col=target_col, frequency=frequency)
            res = sel.select_best_model_parallel(df)
            preds = np.asarray(res.inner.best_model.predict(horizon), dtype=float)
            return res.inner.best_model_name, preds
        return runner

    def run_staged(intra: bool):
        def runner(df: pd.DataFrame) -> tuple[str, np.ndarray]:
            pipe = StagedPipeline(horizon=horizon, timestamp_col=timestamp_col, target_col=target_col,
                                  frequency=frequency, enable_intra_op=intra)
            run = pipe.run(hospital_id="auto_opt", raw_df=df)
            yhats = np.asarray([s["yhat"] for s in run.forecast], dtype=float)
            return run.best_model_name, yhats
        return runner

    return {
        "sequential":    run_sequential,
        "data_thread":   run_data("thread"),
        "data_process":  run_data("process"),
        "staged_inline": run_staged(False),
        "staged_intra":  run_staged(True),
    }


class AutoPipelineOptimizer:
    """Try multiple parallelisation strategies and pick the fastest acceptable one."""

    def __init__(
        self,
        horizon: int = 8,
        timestamp_col: str = "timestamp",
        target_col: str = "icu_occupied",
        frequency: str = "W",
        strategies: tuple[Strategy, ...] = ("sequential", "data_thread", "staged_intra"),
        rmse_tolerance: float = 1.20,
    ) -> None:
        self.horizon = horizon
        self.timestamp_col = timestamp_col
        self.target_col = target_col
        self.frequency = frequency
        self.strategies = strategies
        self.rmse_tolerance = rmse_tolerance

    def explore(self, df: pd.DataFrame, cache_key: str | None = None) -> StrategyDecision:
        if cache_key and cache_key in _DECISION_CACHE:
            return _DECISION_CACHE[cache_key]

        df = df.sort_values(self.timestamp_col).reset_index(drop=True)
        if len(df) <= self.horizon:
            raise ValueError(f"Need > horizon={self.horizon} rows, got {len(df)}")

        train_df = df.iloc[:-self.horizon].copy()
        actual = df[self.target_col].iloc[-self.horizon:].to_numpy()

        trials: List[StrategyTrial] = []
        runners = _trial_runners(
            horizon=self.horizon,
            timestamp_col=self.timestamp_col,
            target_col=self.target_col,
            frequency=self.frequency,
        )

        for strat in self.strategies:
            if strat not in runners:
                continue
            t0 = time.perf_counter()
            try:
                model_name, preds = runners[strat](train_df)
                wall_ms = (time.perf_counter() - t0) * 1000.0
                rmse = _rmse(actual, preds)
                trials.append(StrategyTrial(strategy=strat, wall_ms=wall_ms, best_model=model_name, rmse=rmse))
            except Exception as exc:
                wall_ms = (time.perf_counter() - t0) * 1000.0
                trials.append(StrategyTrial(strategy=strat, wall_ms=wall_ms, best_model="",
                                            rmse=float("inf"), error=str(exc)))

        successes = [t for t in trials if t.is_success() and np.isfinite(t.rmse)]
        if not successes:
            raise RuntimeError("All strategies failed in auto-optimizer.")

        best_rmse = min(t.rmse for t in successes)
        floor = best_rmse * self.rmse_tolerance
        acceptable = [t for t in successes if t.rmse <= floor]
        chosen = min(acceptable, key=lambda t: t.wall_ms)

        decision = StrategyDecision(
            chosen=chosen.strategy,
            chosen_wall_ms=chosen.wall_ms,
            chosen_rmse=chosen.rmse,
            quality_floor_rmse=floor,
            trials=trials,
            explored_strategies=list(self.strategies),
        )
        if cache_key:
            _DECISION_CACHE[cache_key] = decision
        return decision


def reset_decision_cache() -> None:
    _DECISION_CACHE.clear()


__all__ = [
    "AutoPipelineOptimizer",
    "StrategyDecision",
    "StrategyTrial",
    "ALL_STRATEGIES",
    "reset_decision_cache",
]
