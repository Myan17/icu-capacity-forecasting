from __future__ import annotations

import pandas as pd

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:  # pragma: no cover
    SARIMAX = None


class SarimaForecaster:
    """Weekly SARIMA forecaster wrapper."""

    def __init__(
        self,
        target_col: str,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> None:
        self.target_col = target_col
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model = None

    def fit(self, df: pd.DataFrame) -> "SarimaForecaster":
        if SARIMAX is None:
            raise ImportError(
                "statsmodels is not installed. Install it with: pip install statsmodels"
            )

        series = df[self.target_col].astype(float)

        model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.fitted_model = model.fit(disp=False)
        return self

    def predict(self, horizon: int) -> list[float]:
        if self.fitted_model is None:
            raise ValueError("Model must be fit before calling predict")

        forecast = self.fitted_model.forecast(steps=horizon)
        return forecast.tolist()