"""Shared interface for forecasting models.

This is a light-weight base class rather than a strict abstract interface.
The goal is to keep the code easy to follow while encouraging a consistent API.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class ForecastModel(Protocol):
    """Minimal protocol used by the training and inference pipelines."""

    model_name: str

    def fit(self, series: pd.Series) -> None:
        ...

    def forecast(self, horizon: int) -> pd.Series:
        ...
