"""Pure forecast math — no AWS dependencies.

Extracted from handler.py so these can be unit-tested without Lambda context
or DynamoDB. Same pattern as lambdas/ingest/validation_suite.py.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

YELLOW_DEFAULT = 0.75
RED_DEFAULT    = 0.90
MC_SAMPLES     = 500


def breach_prob(
    yhat: float,
    yhat_lower: float,
    yhat_upper: float,
    capacity: int,
    red_threshold: float = RED_DEFAULT,
) -> float:
    """P(occupancy > red_threshold × capacity) using normal approx of the 95% CI.

    sigma = CI_width / 3.92  (inverse of 95% normal CI formula)
    Clamps result to [0, 1].
    """
    sigma = max((yhat_upper - yhat_lower) / 3.92, 0.1)
    red_limit = red_threshold * capacity
    return float(min(max(1.0 - norm.cdf(red_limit, loc=yhat, scale=sigma), 0.0), 1.0))


def risk_label(
    yhat: float,
    capacity: int,
    yellow: float = YELLOW_DEFAULT,
    red: float = RED_DEFAULT,
) -> str:
    """Map predicted occupancy to RED / YELLOW / GREEN."""
    ratio = yhat / max(capacity, 1)
    if ratio >= red:
        return "RED"
    if ratio >= yellow:
        return "YELLOW"
    return "GREEN"


def mc_ci(
    residuals: np.ndarray,
    yhat: float,
    n_samples: int = MC_SAMPLES,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap Monte Carlo 95% confidence interval.

    Samples residuals with replacement, adds them to yhat, returns
    (2.5th percentile, 97.5th percentile).
    """
    if len(residuals) == 0:
        return yhat, yhat
    rng = np.random.default_rng(seed)
    sampled = yhat + rng.choice(residuals, size=n_samples, replace=True)
    return float(np.percentile(sampled, 2.5)), float(np.percentile(sampled, 97.5))
