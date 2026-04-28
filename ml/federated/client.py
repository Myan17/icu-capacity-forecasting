"""Federated learning *client*: a single hospital that trains a local model
without sharing its raw snapshot data.

Following the FedAvg pattern, each client:
  1. receives a global parameter vector ``θ_global``
  2. runs a few local epochs / iterations on its private data
  3. returns its updated parameter vector and the number of training rows

We use a deliberately simple parameter space for the course-project:

    θ = (level, slope, season_strength, baseline_weight)

This is a 4-dimensional vector that controls how the local forecaster is
constructed. It's small enough to make the math obvious but rich enough to
exercise FedAvg / weighted-aggregation logic.

The local forecaster derived from ``θ`` is a Holt-Winters-style additive
model implemented from scratch with NumPy so it has no external dependencies
and parameter aggregation is straightforward.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from shared.hospital_tiers import HospitalProfile


# Parameter index map for readability
_LEVEL, _SLOPE, _SEASON, _BASELINE_W = 0, 1, 2, 3
_PARAM_DIM = 4


def init_global_params() -> np.ndarray:
    """Reasonable starting point: lean on baseline (last value)."""
    theta = np.zeros(_PARAM_DIM, dtype=float)
    theta[_BASELINE_W] = 1.0  # 100 % baseline initially
    return theta


@dataclass
class ClientUpdate:
    hospital_id: str
    theta: np.ndarray
    num_samples: int
    local_loss: float
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hospital_id": self.hospital_id,
            "theta":       self.theta.tolist(),
            "num_samples": self.num_samples,
            "local_loss":  round(self.local_loss, 4),
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class ClientReport:
    hospital_id: str
    history_loss: List[float] = field(default_factory=list)


class FederatedClient:
    """Per-hospital local trainer.

    Parameters
    ----------
    profile : HospitalProfile
        Used both for weighting in aggregation and to decide local epochs.
    target_col, timestamp_col : str
        Column names in the snapshot DataFrame.
    local_epochs : int
        Number of stochastic-gradient passes per round.
    learning_rate : float
        Step size for parameter updates.
    """

    def __init__(
        self,
        profile: HospitalProfile,
        target_col: str = "icu_occupied",
        timestamp_col: str = "timestamp",
        local_epochs: int = 3,
        learning_rate: float = 0.05,
    ) -> None:
        self.profile = profile
        self.target_col = target_col
        self.timestamp_col = timestamp_col
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self._df: pd.DataFrame | None = None
        self.report = ClientReport(hospital_id=profile.hospital_id)

    @property
    def hospital_id(self) -> str:
        return self.profile.hospital_id

    def load_data(self, df: pd.DataFrame) -> "FederatedClient":
        """Hospital-private data; never leaves the client object."""
        self._df = df.sort_values(self.timestamp_col).reset_index(drop=True).copy()
        return self

    def local_train(self, theta_global: np.ndarray) -> ClientUpdate:
        """Run ``local_epochs`` of mini-batch SGD starting from the global params."""
        if self._df is None or self._df.empty:
            return ClientUpdate(
                hospital_id=self.hospital_id,
                theta=theta_global.copy(),
                num_samples=0,
                local_loss=float("inf"),
                duration_ms=0.0,
            )

        y = self._df[self.target_col].astype(float).to_numpy()
        n = len(y)
        if n < 4:
            return ClientUpdate(
                hospital_id=self.hospital_id,
                theta=theta_global.copy(),
                num_samples=n,
                local_loss=float("inf"),
                duration_ms=0.0,
            )

        theta = theta_global.copy()
        t0 = time.perf_counter()
        for _ in range(self.local_epochs):
            preds = _forecast_inplace(theta, y)
            loss = float(np.mean((preds - y) ** 2))
            self.report.history_loss.append(loss)

            grad = _gradient(theta, y, preds)
            theta -= self.learning_rate * grad

        final_preds = _forecast_inplace(theta, y)
        final_loss = float(np.mean((final_preds - y) ** 2))
        return ClientUpdate(
            hospital_id=self.hospital_id,
            theta=theta,
            num_samples=n,
            local_loss=final_loss,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def predict_with_global(self, theta: np.ndarray, horizon: int = 8) -> np.ndarray:
        if self._df is None or self._df.empty:
            return np.zeros(horizon)
        y = self._df[self.target_col].astype(float).to_numpy()
        return _forecast_future(theta, y, horizon)


# ── Local model: Holt-Winters-lite ────────────────────────────────────────────


def _seasonal_index(t: int, period: int = 4) -> float:
    """Cosine seasonality indicator in [-1, 1]; period=4 → quarterly weekly cycle."""
    return float(np.cos(2 * np.pi * (t % period) / period))


def _forecast_inplace(theta: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Predict each in-sample value (used during training)."""
    n = len(y)
    preds = np.empty(n)
    base = float(y[0])
    for t in range(n):
        prev = float(y[t - 1]) if t > 0 else base
        season = _seasonal_index(t)
        drift = theta[_LEVEL] + theta[_SLOPE] * t + theta[_SEASON] * season
        preds[t] = theta[_BASELINE_W] * prev + drift
    return preds


def _forecast_future(theta: np.ndarray, y: np.ndarray, horizon: int) -> np.ndarray:
    """Roll forward `horizon` steps."""
    last = float(y[-1])
    out = np.empty(horizon)
    n = len(y)
    for k in range(horizon):
        t = n + k
        season = _seasonal_index(t)
        drift = theta[_LEVEL] + theta[_SLOPE] * t + theta[_SEASON] * season
        last = theta[_BASELINE_W] * last + drift
        out[k] = last
    return out


def _gradient(theta: np.ndarray, y: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Analytical gradient of mean-squared error wrt θ."""
    n = len(y)
    err = (preds - y)  # shape (n,)
    grad = np.zeros(_PARAM_DIM)

    grad[_LEVEL] = float(np.mean(2.0 * err))
    grad[_SLOPE] = float(np.mean(2.0 * err * np.arange(n)))
    seasonals = np.array([_seasonal_index(t) for t in range(n)])
    grad[_SEASON] = float(np.mean(2.0 * err * seasonals))
    prevs = np.concatenate([[y[0]], y[:-1]])
    grad[_BASELINE_W] = float(np.mean(2.0 * err * prevs)) / max(np.std(prevs), 1.0)

    norm = max(np.linalg.norm(grad), 1.0)  # gradient clipping for stability
    if norm > 1.0:
        grad = grad / norm
    return grad


__all__ = [
    "FederatedClient",
    "ClientUpdate",
    "ClientReport",
    "init_global_params",
]
