"""Per-hospital model hyperparameter tuner — Lecture 4 data heterogeneity.

The fleet contains hospitals with very different time-series characteristics:
some are noisy (small ICUs with high week-to-week variance), some are stable
academic medical centres. A single global hyperparameter set is sub-optimal —
this is the "non-IID" problem federated-learning papers obsess over.

We do a small grid search per hospital over the most impactful knobs:

  * Prophet : ``changepoint_prior_scale`` (0.01 / 0.05 / 0.5)
  * SARIMA  : ``order`` ((1,1,1) / (2,1,2) / (1,1,2))

Cost-aware grid: small (Tier 3) hospitals get 1 candidate, medium 2, large 3.
This keeps the per-hospital tuning budget bounded but still allocates more
search to data-rich hospitals (mirrors TiFL's resource-aware design).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ml.models.baseline import BaselineForecaster
from ml.models.prophet_model import ProphetForecaster
from ml.models.sarima_model import SarimaForecaster
from shared.hospital_tiers import HospitalProfile, WorkloadTier


@dataclass
class TunedConfig:
    hospital_id: str
    model_name: str
    params: Dict[str, Any]
    rmse: float
    mae: float
    explored: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hospital_id": self.hospital_id,
            "model_name":  self.model_name,
            "params":      self.params,
            "rmse":        round(self.rmse, 4),
            "mae":         round(self.mae, 4),
            "explored":    self.explored,
        }


PROPHET_GRID = [
    {"changepoint_prior_scale": 0.05},
    {"changepoint_prior_scale": 0.5},
    {"changepoint_prior_scale": 0.01},
]

SARIMA_GRID = [
    {"order": (1, 1, 1)},
    {"order": (2, 1, 2)},
    {"order": (1, 1, 2)},
]


_TIER_BUDGET: Dict[WorkloadTier, int] = {
    WorkloadTier.SMALL: 1,
    WorkloadTier.MEDIUM: 2,
    WorkloadTier.LARGE: 3,
}


class PerHospitalTuner:
    def __init__(
        self,
        horizon: int = 8,
        timestamp_col: str = "timestamp",
        target_col: str = "icu_occupied",
        frequency: str = "W",
    ) -> None:
        self.horizon = horizon
        self.timestamp_col = timestamp_col
        self.target_col = target_col
        self.frequency = frequency

    def tune(self, profile: HospitalProfile, df: pd.DataFrame) -> TunedConfig:
        df = df.sort_values(self.timestamp_col).reset_index(drop=True)
        if len(df) <= self.horizon:
            return TunedConfig(
                hospital_id=profile.hospital_id,
                model_name="baseline",
                params={},
                rmse=float("inf"),
                mae=float("inf"),
                explored=[],
            )

        train_df = df.iloc[:-self.horizon].copy()
        valid_actual = df[self.target_col].iloc[-self.horizon:].to_numpy()

        budget = _TIER_BUDGET[profile.workload_tier]
        candidates = self._candidate_grid(budget)

        explored: List[Dict[str, Any]] = []
        best: TunedConfig | None = None

        for spec in candidates:
            model_name = spec["model"]
            params = spec["params"]
            try:
                t0 = time.perf_counter()
                preds = self._fit_predict(model_name, params, train_df)
                duration_ms = (time.perf_counter() - t0) * 1000.0
                rmse, mae = self._score(valid_actual, preds)
                trial = {
                    "model_name": model_name,
                    "params":     params,
                    "rmse":       round(rmse, 4),
                    "mae":        round(mae, 4),
                    "duration_ms": round(duration_ms, 2),
                    "status":     "success",
                }
                explored.append(trial)
                if best is None or rmse < best.rmse:
                    best = TunedConfig(
                        hospital_id=profile.hospital_id,
                        model_name=model_name,
                        params=params,
                        rmse=rmse,
                        mae=mae,
                        explored=explored,
                    )
            except Exception as exc:
                explored.append({
                    "model_name": model_name,
                    "params":     params,
                    "status":     "failed",
                    "error":      str(exc),
                })

        if best is None:
            best = TunedConfig(
                hospital_id=profile.hospital_id,
                model_name="baseline",
                params={},
                rmse=float("inf"),
                mae=float("inf"),
                explored=explored,
            )
        return best

    # ── helpers ────────────────────────────────────────────────────────────────

    def _candidate_grid(self, budget: int) -> List[Dict[str, Any]]:
        """Always include baseline; then add Prophet/SARIMA up to the budget."""
        grid: List[Dict[str, Any]] = [{"model": "baseline", "params": {}}]
        prophet_slice = PROPHET_GRID[:budget]
        sarima_slice = SARIMA_GRID[:budget]
        for params in prophet_slice:
            grid.append({"model": "prophet", "params": dict(params)})
        for params in sarima_slice:
            grid.append({"model": "sarima", "params": dict(params)})
        return grid

    def _fit_predict(self, model_name: str, params: Dict[str, Any], train_df: pd.DataFrame) -> np.ndarray:
        if model_name == "baseline":
            m = BaselineForecaster(target_col=self.target_col)
            m.fit(train_df)
        elif model_name == "prophet":
            m = ProphetForecaster(target_col=self.target_col, frequency=self.frequency)
            m.fit(train_df, timestamp_col=self.timestamp_col)
            if hasattr(m, "model") and m.model is not None and "changepoint_prior_scale" in params:
                # Re-init with the desired hyperparameter — Prophet doesn't accept
                # changes after fit, so rebuild from scratch.
                from prophet import Prophet  # type: ignore
                m.model = Prophet(
                    daily_seasonality=False,
                    weekly_seasonality=False,
                    yearly_seasonality=True,
                    changepoint_prior_scale=params["changepoint_prior_scale"],
                )
                tdf = train_df[[self.timestamp_col, self.target_col]].rename(
                    columns={self.timestamp_col: "ds", self.target_col: "y"}
                )
                m.model.fit(tdf)
        elif model_name == "sarima":
            order = params.get("order", (1, 1, 1))
            m = SarimaForecaster(target_col=self.target_col, order=order)
            m.fit(train_df)
        else:
            raise ValueError(f"unknown model: {model_name}")
        return np.asarray(m.predict(self.horizon), dtype=float)

    def _score(self, actual: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
        n = min(len(actual), len(pred))
        if n == 0:
            return float("inf"), float("inf")
        a, p = actual[:n], pred[:n]
        rmse = float(np.sqrt(np.mean((a - p) ** 2)))
        mae = float(np.mean(np.abs(a - p)))
        return rmse, mae


def tune_hospital(
    profile: HospitalProfile,
    df: pd.DataFrame,
    horizon: int = 8,
) -> TunedConfig:
    return PerHospitalTuner(horizon=horizon).tune(profile, df)


__all__ = ["PerHospitalTuner", "TunedConfig", "tune_hospital", "PROPHET_GRID", "SARIMA_GRID"]
