from __future__ import annotations

import pandas as pd


class BaselineForecaster:
    """
    Simple last-value forecaster.
    Predicts the last observed target value for all future periods.
    """

    def __init__(self, target_col: str) -> None:
        self.target_col = target_col
        self.last_value: float | None = None

    def fit(self, df: pd.DataFrame) -> "BaselineForecaster":
        if df.empty:
            raise ValueError("Cannot fit baseline model on empty dataframe")

        self.last_value = float(df[self.target_col].dropna().iloc[-1])
        return self

    def predict(self, horizon: int) -> list[float]:
        if self.last_value is None:
            raise ValueError("Model must be fit before calling predict")

        return [self.last_value] * horizon