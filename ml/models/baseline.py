"""Simple baseline forecasting strategies."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd


class BaselineForecaster:
    """Forecast using simple, interpretable heuristics.

    Supported strategies:
    - rolling_mean
    - last_value
    """

    model_name = "baseline"

    def __init__(self, rolling_window: int = 6, strategy: str = "rolling_mean") -> None:
        self.rolling_window = rolling_window
        self.strategy = strategy
        self.history: pd.Series | None = None
        self.last_prediction: float | None = None

    def fit(self, series: pd.Series) -> None:
        """Fit the baseline model by storing summary information from history."""
        clean_series = series.dropna().astype(float)
        if clean_series.empty:
            raise ValueError("Cannot fit baseline model on an empty series")

        self.history = clean_series

        if self.strategy == "last_value":
            self.last_prediction = float(clean_series.iloc[-1])
        elif self.strategy == "rolling_mean":
            window = min(self.rolling_window, len(clean_series))
            self.last_prediction = float(clean_series.tail(window).mean())
        else:
            raise ValueError(f"Unsupported baseline strategy: {self.strategy}")

    def forecast(self, horizon: int) -> pd.Series:
        """Repeat the baseline estimate for each future step."""
        if horizon <= 0:
            raise ValueError("Forecast horizon must be positive")
        if self.last_prediction is None:
            raise ValueError("Model must be fit before forecasting")
        return pd.Series([self.last_prediction] * horizon, dtype="float64")

    def save(self, path: str | Path) -> None:
        """Serialize the fitted baseline model."""
        if self.last_prediction is None:
            raise ValueError("Cannot save an unfitted model")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file_obj:
            pickle.dump(self, file_obj)

    @classmethod
    def load(cls, path: str | Path) -> "BaselineForecaster":
        """Load a serialized baseline model."""
        input_path = Path(path)
        with input_path.open("rb") as file_obj:
            model = pickle.load(file_obj)

        if not isinstance(model, cls):
            raise TypeError(f"Serialized object is not a {cls.__name__}")
        return model