"""Prophet forecasting wrapper."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd


class ProphetForecaster:
    """Wrapper that exposes a fit/forecast interface similar to other models."""

    model_name = "prophet"

    def __init__(
        self,
        daily_seasonality: bool = True,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = False,
        frequency: str = "1h",
    ) -> None:
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.frequency = frequency.lower()
        self._model = None
        self._training_frame: pd.DataFrame | None = None

    def fit(self, series: pd.Series) -> None:
        """Fit Prophet on a pandas Series with a DatetimeIndex."""
        try:
            from prophet import Prophet
        except ImportError as exc:
            raise ImportError(
                "prophet is required for the Prophet model. "
                "Install it with: pip install prophet"
            ) from exc

        clean_series = series.dropna().astype(float)
        if clean_series.empty:
            raise ValueError("Cannot fit Prophet model on an empty series")
        if not isinstance(clean_series.index, pd.DatetimeIndex):
            raise ValueError("Prophet requires a DatetimeIndex on the training series")

        # Remove duplicated timestamps defensively.
        clean_series = clean_series[~clean_series.index.duplicated(keep="last")]

        training_frame = pd.DataFrame(
            {"ds": clean_series.index, "y": clean_series.values}
        ).dropna()

        if training_frame.empty:
            raise ValueError("No valid rows remain after Prophet preprocessing")

        model = Prophet(
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
        )
        model.fit(training_frame)

        self._model = model
        self._training_frame = training_frame

    def forecast(self, horizon: int) -> pd.Series:
        """Produce future predictions for the requested horizon."""
        if horizon <= 0:
            raise ValueError("Forecast horizon must be positive")
        if self._model is None or self._training_frame is None:
            raise ValueError("Model must be fit before forecasting")

        future = self._model.make_future_dataframe(
            periods=horizon,
            freq=self.frequency,
            include_history=False,
        )
        forecast_frame = self._model.predict(future)
        return pd.Series(
            forecast_frame["yhat"].values,
            index=future["ds"],
            dtype="float64",
        )

    def save(self, path: str | Path) -> None:
        """Serialize the fitted Prophet wrapper."""
        if self._model is None or self._training_frame is None:
            raise ValueError("Cannot save an unfitted model")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file_obj:
            pickle.dump(self, file_obj)

    @classmethod
    def load(cls, path: str | Path) -> "ProphetForecaster":
        """Load a serialized Prophet wrapper."""
        input_path = Path(path)
        with input_path.open("rb") as file_obj:
            model = pickle.load(file_obj)

        if not isinstance(model, cls):
            raise TypeError(f"Serialized object is not a {cls.__name__}")
        return model