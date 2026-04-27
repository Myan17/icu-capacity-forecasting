"""Unit tests for the MLForecastService.

These tests exercise the service in isolation using synthetic data,
without requiring a running database or backend server.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path for ml.* and lambdas.* imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "lambdas")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import pytest

from app.services.forecast import ForecastStep, MLForecastService


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(n: int = 20) -> pd.DataFrame:
    """Create a simple synthetic hospital time series with n weekly rows."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-07", periods=n, freq="W"),
        "icu_occupied": [60 + (i % 10) for i in range(n)],
        "icu_capacity": [100] * n,
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_baseline_forecast_returns_correct_horizon():
    service = MLForecastService(model_preference="baseline", horizon=6)
    steps = service.generate_forecast(_make_df(), capacity=100)
    assert len(steps) == 6


def test_forecast_step_has_all_fields():
    service = MLForecastService(model_preference="baseline", horizon=4)
    steps = service.generate_forecast(_make_df(), capacity=100)
    for step in steps:
        assert isinstance(step, ForecastStep)
        assert step.yhat_lower is not None
        assert step.yhat_upper is not None
        assert step.breach_prob is not None
        assert step.risk_level in ("GREEN", "YELLOW", "RED")
        assert step.model_name == "baseline"
        assert step.forecast_time is not None


def test_ci_lower_leq_upper():
    service = MLForecastService(model_preference="baseline", horizon=4)
    steps = service.generate_forecast(_make_df(), capacity=100)
    for step in steps:
        # The real invariant: lower bound is below upper bound
        assert step.yhat_lower <= step.yhat_upper


def test_breach_prob_bounded():
    service = MLForecastService(model_preference="baseline", horizon=4)
    steps = service.generate_forecast(_make_df(), capacity=100)
    for step in steps:
        assert 0.0 <= step.breach_prob <= 1.0


def test_risk_label_consistency():
    """When occupancy is well below thresholds, risk should be GREEN."""
    # All values around 60, capacity 100 → ratio ~0.60 → GREEN
    service = MLForecastService(model_preference="baseline", horizon=4)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-07", periods=20, freq="W"),
        "icu_occupied": [60] * 20,
        "icu_capacity": [100] * 20,
    })
    steps = service.generate_forecast(df, capacity=100)
    for step in steps:
        assert step.risk_level == "GREEN"


def test_auto_selects_best_model():
    service = MLForecastService(model_preference="auto", horizon=4)
    steps = service.generate_forecast(_make_df(), capacity=100)
    assert len(steps) > 0
    model_name = steps[0].model_name
    assert model_name in ("baseline", "sarima", "prophet")


def test_explicit_sarima_model():
    service = MLForecastService(model_preference="sarima", horizon=4)
    steps = service.generate_forecast(_make_df(), capacity=100)
    assert len(steps) == 4
    for step in steps:
        assert step.model_name == "sarima"


def test_insufficient_data_raises():
    service = MLForecastService(model_preference="baseline", horizon=4)
    tiny_df = _make_df(n=2)
    with pytest.raises(ValueError, match="at least"):
        service.generate_forecast(tiny_df, capacity=100)
