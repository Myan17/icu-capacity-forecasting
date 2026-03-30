"""SARIMA forecasting wrapper.

The import is intentionally lazy so the rest of the codebase can still run even
if `statsmodels` is not installed yet.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd


class SarimaForecaster:
    """Thin wrapper around statsmodels SARIMAX."""

    model_name = "sarima"

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self._fitted_model = None

    def fit(self, series: pd.Series) -> None:
        """Fit a SARIMA model to the provided series."""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError as exc:
            raise ImportError(
                "statsmodels is required for the SARIMA model. "
                "Install it with: pip install statsmodels"
            ) from exc

        clean_series = series.dropna().astype(float)
        if clean_series.empty:
            raise ValueError("Cannot fit SARIMA model on an empty series")

        if not isinstance(clean_series.index, pd.DatetimeIndex):
            raise ValueError("SARIMA requires a DatetimeIndex on the training series")

        # Fallback for short histories: avoid seasonal fitting when there is
        # clearly not enough data to support it.
        seasonal_order = self.seasonal_order
        season_length = seasonal_order[-1]
        if len(clean_series) < max(2 * season_length, 24):
            seasonal_order = (0, 0, 0, 0)

        model = SARIMAX(
            clean_series,
            order=self.order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._fitted_model = model.fit(disp=False)
        self.seasonal_order = seasonal_order

    def forecast(self, horizon: int) -> pd.Series:
        """Forecast the requested number of future time steps."""
        if horizon <= 0:
            raise ValueError("Forecast horizon must be positive")
        if self._fitted_model is None:
            raise ValueError("Model must be fit before forecasting")

        forecast = self._fitted_model.forecast(steps=horizon)
        return pd.Series(forecast, dtype="float64")

    def save(self, path: str | Path) -> None:
        """Serialize the fitted SARIMA model."""
        if self._fitted_model is None:
            raise ValueError("Cannot save an unfitted model")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file_obj:
            pickle.dump(self, file_obj)

    @classmethod
    def load(cls, path: str | Path) -> "SarimaForecaster":
        """Load a serialized SARIMA model."""
        input_path = Path(path)
        with input_path.open("rb") as file_obj:
            model = pickle.load(file_obj)

        if not isinstance(model, cls):
            raise TypeError(f"Serialized object is not a {cls.__name__}")
        return model