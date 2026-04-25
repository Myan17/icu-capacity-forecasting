"""Unit tests for lambdas/forecast/forecast_utils.py — no AWS, no DynamoDB."""
from __future__ import annotations

import numpy as np
import pytest

from lambdas.forecast.forecast_utils import breach_prob, mc_ci, risk_label


# ── breach_prob ───────────────────────────────────────────────────────────────

def test_breach_prob_certain():
    # yhat (100) well above 90% of capacity (100) → P should be > 0.9
    p = breach_prob(yhat=100, yhat_lower=95, yhat_upper=105, capacity=100)
    assert p > 0.9


def test_breach_prob_no_breach():
    # yhat (50) well below 90% of capacity (100) → P should be < 0.1
    p = breach_prob(yhat=50, yhat_lower=45, yhat_upper=55, capacity=100)
    assert p < 0.1


def test_breach_prob_bounded():
    p = breach_prob(yhat=200, yhat_lower=190, yhat_upper=210, capacity=100)
    assert 0.0 <= p <= 1.0


def test_breach_prob_wide_ci_raises_probability():
    # Same yhat, wider CI → more mass past threshold
    narrow = breach_prob(yhat=85, yhat_lower=84, yhat_upper=86, capacity=100)
    wide   = breach_prob(yhat=85, yhat_lower=70, yhat_upper=100, capacity=100)
    assert wide > narrow


# ── risk_label ────────────────────────────────────────────────────────────────

def test_risk_label_green():
    assert risk_label(70, 100) == "GREEN"


def test_risk_label_yellow():
    assert risk_label(80, 100) == "YELLOW"


def test_risk_label_red():
    assert risk_label(92, 100) == "RED"


def test_risk_label_at_yellow_boundary():
    # Exactly 75% capacity → YELLOW (threshold is >=)
    assert risk_label(75, 100) == "YELLOW"


# ── mc_ci ─────────────────────────────────────────────────────────────────────

def test_mc_ci_interval_contains_yhat():
    residuals = np.zeros(100)  # zero residuals → CI collapses to yhat
    lo, hi = mc_ci(residuals, yhat=80.0, seed=42)
    assert lo <= 80.0 <= hi


def test_mc_ci_wider_with_spread():
    tight = np.random.default_rng(0).normal(0, 1, 200)
    wide  = np.random.default_rng(0).normal(0, 20, 200)
    lo_t, hi_t = mc_ci(tight, yhat=80.0)
    lo_w, hi_w = mc_ci(wide,  yhat=80.0)
    assert (hi_w - lo_w) > (hi_t - lo_t)


def test_mc_ci_empty_residuals_returns_yhat():
    lo, hi = mc_ci(np.array([]), yhat=50.0)
    assert lo == 50.0
    assert hi == 50.0
