"""ML-powered forecast service for the live API.

Replaces the naive rolling-mean forecast with real ML models (Baseline,
SARIMA, Prophet) from the ml/ module.  Produces confidence intervals via
Monte Carlo bootstrapping and computes breach probabilities.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

# Ensure ml.* and lambdas.* are importable
import app.services.ml_imports  # noqa: F401

import numpy as np
import pandas as pd

from ml.models.baseline import BaselineForecaster
from ml.models.sarima_model import SarimaForecaster

try:
    from ml.models.prophet_model import ProphetForecaster
    _PROPHET_AVAILABLE = True
except ImportError:
    ProphetForecaster = None  # type: ignore[assignment,misc]
    _PROPHET_AVAILABLE = False

from forecast.forecast_utils import breach_prob, mc_ci, risk_label

logger = logging.getLogger(__name__)

# Minimum number of data points needed to fit a model
MIN_DATA_POINTS = 4


@dataclass
class ForecastStep:
    """One time-step of a forecast, ready to be persisted."""
    forecast_time: datetime
    predicted_icu_occupied: float
    yhat_lower: float
    yhat_upper: float
    risk_level: str
    model_name: str
    breach_prob: float


class MLForecastService:
    """Fit an ML model on snapshot history and generate forecasts.

    Parameters
    ----------
    model_preference : str
        One of ``"auto"``, ``"baseline"``, ``"sarima"``, ``"prophet"``.
        ``"auto"`` tries all available models on a train/validation split
        and picks the one with the lowest RMSE.
    horizon : int
        Number of forecast steps (weeks) to generate.
    frequency : str
        Pandas offset alias for the time series frequency.
    yellow_threshold : float
        Occupancy ratio threshold for YELLOW risk.
    red_threshold : float
        Occupancy ratio threshold for RED risk.
    """

    def __init__(
        self,
        model_preference: str = "auto",
        horizon: int = 8,
        frequency: str = "W",
        yellow_threshold: float = 0.75,
        red_threshold: float = 0.90,
    ) -> None:
        self.model_preference = model_preference
        self.horizon = horizon
        self.frequency = frequency
        self.yellow_threshold = yellow_threshold
        self.red_threshold = red_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_forecast(
        self,
        df: pd.DataFrame,
        capacity: int,
    ) -> list[ForecastStep]:
        """Generate a full forecast from snapshot history.

        Parameters
        ----------
        df : DataFrame
            Must contain at least ``timestamp`` and ``icu_occupied`` columns,
            sorted ascending by timestamp.
        capacity : int
            Current ICU capacity for the hospital.

        Returns
        -------
        list[ForecastStep]
            One entry per forecast step with CI, risk, and breach probability.

        Raises
        ------
        ValueError
            If there are too few data points to fit a model.
        """
        if len(df) < MIN_DATA_POINTS:
            raise ValueError(
                f"Need at least {MIN_DATA_POINTS} data points, got {len(df)}"
            )

        model, model_name = self._select_and_fit(df)
        predictions = self._predict(model, self.horizon)

        # Compute residuals for Monte Carlo CI
        residuals = self._compute_residuals(model, model_name, df)

        now = datetime.now(timezone.utc)
        steps: list[ForecastStep] = []

        for i, yhat in enumerate(predictions):
            lo, hi = mc_ci(residuals, float(yhat), seed=42 + i)
            risk = risk_label(
                float(yhat),
                capacity,
                yellow=self.yellow_threshold,
                red=self.red_threshold,
            )
            bp = breach_prob(
                yhat=float(yhat),
                yhat_lower=lo,
                yhat_upper=hi,
                capacity=capacity,
                red_threshold=self.red_threshold,
            )
            steps.append(ForecastStep(
                forecast_time=now + timedelta(weeks=i + 1),
                predicted_icu_occupied=float(yhat),
                yhat_lower=lo,
                yhat_upper=hi,
                risk_level=risk,
                model_name=model_name,
                breach_prob=round(bp, 4),
            ))

        return steps

    # ------------------------------------------------------------------
    # Model selection / fitting
    # ------------------------------------------------------------------

    def _select_and_fit(self, df: pd.DataFrame) -> tuple[Any, str]:
        """Select and fit the requested model.

        When ``model_preference`` is ``"auto"``, all available models are
        evaluated on a train/validation split and the best one (lowest RMSE)
        is retrained on the full dataset.
        """
        if self.model_preference == "auto":
            return self._auto_select(df)

        model = self._build_model(self.model_preference)
        self._fit_model(model, df)
        return model, self.model_preference

    def _auto_select(self, df: pd.DataFrame) -> tuple[Any, str]:
        """Try all candidate models and pick the best by RMSE."""
        val_size = min(self.horizon, len(df) // 3)
        if val_size < 2:
            # Not enough data for validation — fall back to baseline
            logger.warning("Not enough data for auto-selection, falling back to baseline")
            model = BaselineForecaster(target_col="icu_occupied")
            model.fit(df)
            return model, "baseline"

        train_df = df.iloc[:-val_size].copy()
        val_df = df.iloc[-val_size:].copy()
        actuals = val_df["icu_occupied"].values.astype(float)

        candidates = self._get_candidates()
        best_model = None
        best_name = "baseline"
        best_rmse = float("inf")

        for name, factory in candidates.items():
            try:
                model = factory()
                self._fit_model(model, train_df)
                preds = np.array(self._predict(model, val_size), dtype=float)
                k = min(len(preds), len(actuals))
                rmse = float(np.sqrt(np.mean((actuals[:k] - preds[:k]) ** 2)))
                logger.info("auto-select: %s RMSE=%.3f", name, rmse)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_name = name
                    best_model = model
            except Exception as exc:
                logger.warning("auto-select: %s failed: %s", name, exc)

        if best_model is None:
            raise RuntimeError("All candidate models failed during auto-selection")

        # Retrain best model on full dataset
        final_model = candidates[best_name]()
        self._fit_model(final_model, df)
        logger.info("auto-select: chose %s (RMSE=%.3f), retrained on full data", best_name, best_rmse)
        return final_model, best_name

    def _get_candidates(self) -> dict[str, Any]:
        """Return a dict of model name → factory callable."""
        candidates: dict[str, Any] = {
            "baseline": lambda: BaselineForecaster(target_col="icu_occupied"),
            "sarima": lambda: SarimaForecaster(target_col="icu_occupied"),
        }
        if _PROPHET_AVAILABLE:
            candidates["prophet"] = lambda: ProphetForecaster(
                target_col="icu_occupied", frequency=self.frequency,
            )
        return candidates

    def _build_model(self, model_name: str) -> Any:
        """Instantiate a single model by name."""
        if model_name == "baseline":
            return BaselineForecaster(target_col="icu_occupied")
        if model_name == "sarima":
            return SarimaForecaster(target_col="icu_occupied")
        if model_name == "prophet":
            if not _PROPHET_AVAILABLE:
                raise ImportError(
                    "Prophet is not installed. Falling back is not supported "
                    "when an explicit model is requested."
                )
            return ProphetForecaster(
                target_col="icu_occupied", frequency=self.frequency,
            )
        raise ValueError(f"Unknown model: {model_name!r}")

    def _fit_model(self, model: Any, df: pd.DataFrame) -> None:
        """Fit a model, handling Prophet's different signature."""
        if _PROPHET_AVAILABLE and isinstance(model, ProphetForecaster):
            model.fit(df, timestamp_col="timestamp")
        else:
            model.fit(df)

    def _predict(self, model: Any, horizon: int) -> list[float]:
        """Get predictions from a fitted model as a flat list of floats."""
        raw = model.predict(horizon)
        if isinstance(raw, pd.DataFrame):
            col = "yhat" if "yhat" in raw.columns else raw.select_dtypes("number").columns[0]
            return raw[col].tolist()
        if isinstance(raw, (pd.Series, np.ndarray)):
            return list(np.asarray(raw, dtype=float))
        return list(raw)

    # ------------------------------------------------------------------
    # Residual computation for MC confidence intervals
    # ------------------------------------------------------------------

    def _compute_residuals(
        self,
        model: Any,
        model_name: str,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """Compute in-sample residuals for Monte Carlo CI.

        Uses a simple backtest: fit on the first 80% of data, predict
        on the last 20%, compute residuals.
        """
        series = df["icu_occupied"].values.astype(float)
        split = max(len(series) - self.horizon, int(len(series) * 0.8))

        if split < MIN_DATA_POINTS:
            # Not enough data to compute residuals — return small default noise
            return np.array([0.0])

        train_df = df.iloc[:split].copy()
        actuals = series[split:]

        try:
            backtest_model = self._build_model(model_name)
            self._fit_model(backtest_model, train_df)
            preds = np.array(self._predict(backtest_model, len(actuals)), dtype=float)
            k = min(len(preds), len(actuals))
            return actuals[:k] - preds[:k]
        except Exception as exc:
            logger.warning("Residual computation failed: %s. Using zero residuals.", exc)
            return np.array([0.0])