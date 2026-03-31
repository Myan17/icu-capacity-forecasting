from __future__ import annotations

import pandas as pd

try:
    from prophet import Prophet
except ImportError:  # pragma: no cover
    Prophet = None


class ProphetForecaster:
    """Weekly Prophet forecaster wrapper."""

    def __init__(self, target_col: str, frequency: str = "W") -> None:
        self.target_col = target_col
        self.frequency = frequency
        self.model = None

    def fit(self, df: pd.DataFrame, timestamp_col: str) -> "ProphetForecaster":
        if Prophet is None:
            raise ImportError("Prophet is not installed. Install it with: pip install prophet")

        train_df = df[[timestamp_col, self.target_col]].copy()
        train_df = train_df.rename(columns={timestamp_col: "ds", self.target_col: "y"})

        self.model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        self.model.fit(train_df)
        return self

    def predict(self, horizon: int) -> list[float]:
        if self.model is None:
            raise ValueError("Model must be fit before calling predict")

        future = self.model.make_future_dataframe(periods=horizon, freq=self.frequency)
        forecast = self.model.predict(future)
        return forecast["yhat"].tail(horizon).tolist()