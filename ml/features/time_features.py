"""Optional time-based features for analytics and future multivariate models."""

from __future__ import annotations

import pandas as pd

from ml.config import DEFAULT_CONFIG, ForecastConfig


def add_time_features(
    df: pd.DataFrame,
    config: ForecastConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Add lightweight calendar features."""
    enriched = df.copy()
    ts = pd.to_datetime(enriched[config.timestamp_column], errors="coerce")

    enriched["hour_of_day"] = ts.dt.hour
    enriched["day_of_week"] = ts.dt.dayofweek
    enriched["is_weekend"] = ts.dt.dayofweek >= 5
    enriched["month"] = ts.dt.month
    enriched["day_of_month"] = ts.dt.day
    return enriched


def add_lag_features(
    df: pd.DataFrame,
    lags: list[int],
    config: ForecastConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Add lagged versions of the target column per hospital."""
    enriched = df.copy()
    enriched = enriched.sort_values(
        [config.hospital_id_column, config.timestamp_column],
        kind="mergesort",
    ).copy()

    for lag in lags:
        enriched[f"{config.target_column}_lag_{lag}"] = (
            enriched.groupby(config.hospital_id_column)[config.target_column]
            .shift(lag)
        )

    return enriched


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int],
    config: ForecastConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Add rolling mean features per hospital."""
    enriched = df.copy()
    enriched = enriched.sort_values(
        [config.hospital_id_column, config.timestamp_column],
        kind="mergesort",
    ).copy()

    grouped = enriched.groupby(config.hospital_id_column)[config.target_column]
    for window in windows:
        enriched[f"{config.target_column}_rolling_mean_{window}"] = (
            grouped.transform(lambda s: s.rolling(window=window, min_periods=1).mean())
        )

    return enriched