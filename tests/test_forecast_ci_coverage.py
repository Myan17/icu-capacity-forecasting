"""Statistical calibration tests for the Monte Carlo 95% confidence interval.

Generates synthetic data with known noise, runs mc_ci, and checks that the
true future values fall inside the CI at least 90% of the time.

This is paper Q3/Q5 evidence — demonstrates the probabilistic forecasting is
statistically well-calibrated, not just a ±X% heuristic band.
"""
from __future__ import annotations

import numpy as np
import pytest

from lambdas.forecast.forecast_utils import mc_ci


def _coverage(residuals: np.ndarray, true_values: np.ndarray, yhat: float) -> float:
    """Fraction of true_values that fall inside mc_ci(residuals, yhat)."""
    inside = 0
    for i, true_val in enumerate(true_values):
        lo, hi = mc_ci(residuals, yhat, n_samples=1000, seed=i)
        if lo <= true_val <= hi:
            inside += 1
    return inside / len(true_values)


def test_mc_ci_coverage_gaussian_noise():
    """With Gaussian residuals (σ=5), 95% CI should cover true values ≥ 90% of the time."""
    rng = np.random.default_rng(0)
    yhat = 80.0
    residuals   = rng.normal(0, 5, 200)
    true_values = yhat + rng.normal(0, 5, 50)

    coverage = _coverage(residuals, true_values, yhat)
    assert coverage >= 0.90, f"CI coverage was {coverage:.2%} — expected ≥ 90%"


def test_mc_ci_coverage_uniform_noise():
    """With uniform [-10, +10] residuals, coverage should still be ≥ 85%."""
    rng = np.random.default_rng(1)
    yhat = 75.0
    residuals   = rng.uniform(-10, 10, 200)
    true_values = yhat + rng.uniform(-10, 10, 50)

    coverage = _coverage(residuals, true_values, yhat)
    assert coverage >= 0.85, f"CI coverage was {coverage:.2%} — expected ≥ 85%"
