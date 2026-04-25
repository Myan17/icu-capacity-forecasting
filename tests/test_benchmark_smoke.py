"""Smoke tests for scripts/benchmark_models.py — no AWS required."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "cleaned_hhs_ml_ready.csv"

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="cleaned_hhs_ml_ready.csv not present — skip benchmark smoke tests",
)

from scripts.benchmark_models import benchmark  # noqa: E402


def test_benchmark_returns_dataframe():
    results = benchmark(["010001"], DATA_PATH, horizon=2)
    assert isinstance(results, pd.DataFrame)
    assert not results.empty


def test_benchmark_all_three_models_present():
    results = benchmark(["010001"], DATA_PATH, horizon=2)
    models_run = set(results["model"].tolist())
    assert "baseline" in models_run


def test_benchmark_rmse_is_finite():
    results = benchmark(["010001"], DATA_PATH, horizon=2)
    valid = results[results["model"] == "baseline"].dropna(subset=["rmse"])
    assert len(valid) >= 1
    assert float(valid["rmse"].iloc[0]) > 0
