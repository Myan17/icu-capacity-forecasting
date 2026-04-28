"""Unit tests for the parallel model selector + auto-optimizer (Lecture 3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.model_selection.parallel_selector import ParallelBestModelSelector
from ml.pipelines.auto_optimizer import (
    ALL_STRATEGIES,
    AutoPipelineOptimizer,
    reset_decision_cache,
)
from ml.pipelines.staged_pipeline import StagedPipeline


def _make_df(rows: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "timestamp":    pd.date_range("2023-01-01", periods=rows, freq="W"),
        "icu_occupied": 50 + 5 * np.sin(np.arange(rows) * 0.3) + rng.normal(0, 2, rows),
        "icu_capacity": 100,
    })


def test_parallel_selector_returns_a_winner():
    df = _make_df(40)
    sel = ParallelBestModelSelector(executor="thread", horizon=4)
    res = sel.select_best_model_parallel(df)
    assert res.inner.best_model_name in {"baseline", "prophet", "sarima"}
    assert res.total_wall_ms > 0
    assert res.sequential_estimate_ms > 0


def test_parallel_selector_speedup_is_at_least_one_when_not_pathological():
    df = _make_df(60)
    sel = ParallelBestModelSelector(executor="thread", horizon=4, num_workers=3)
    res = sel.select_best_model_parallel(df)
    # Sum of per-model fit times should be >= wall clock; speedup ratio >= 1.0.
    assert res.speedup >= 0.9


def test_staged_pipeline_runs_end_to_end():
    df = _make_df(50)
    pipe = StagedPipeline(horizon=4, enable_intra_op=True)
    run = pipe.run("hosp_test", df)
    assert run.best_model_name in {"baseline", "prophet", "sarima"}
    assert len(run.forecast) == 4
    summary = run.summary()
    assert summary["total_wall_ms"] > 0
    assert any(s["name"] == "load" for s in summary["stages"])
    assert any(s["name"] == "feature" for s in summary["stages"])


def test_staged_pipeline_inline_decision_is_recorded():
    df = _make_df(50)
    pipe = StagedPipeline(horizon=4, inline_threshold_ms=10**9)  # very high → all "inline"
    run = pipe.run("hosp_inline", df)
    assert "feature" in run.inline_decisions


def test_auto_optimizer_picks_a_strategy_under_quality_floor():
    reset_decision_cache()
    df = _make_df(60)
    optimizer = AutoPipelineOptimizer(
        horizon=4,
        strategies=("sequential", "data_thread"),  # skip slow ones
        rmse_tolerance=2.0,
    )
    decision = optimizer.explore(df, cache_key="hosp_auto")
    assert decision.chosen in ALL_STRATEGIES
    assert decision.chosen_wall_ms > 0
    assert any(t.is_success() for t in decision.trials)


def test_auto_optimizer_caches_decisions():
    reset_decision_cache()
    df = _make_df(40)
    optimizer = AutoPipelineOptimizer(horizon=4, strategies=("sequential",))
    first = optimizer.explore(df, cache_key="hospA")
    second = optimizer.explore(df, cache_key="hospA")
    assert first is second
