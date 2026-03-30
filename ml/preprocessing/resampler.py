"""Resampling helpers for hospital ICU time-series data."""

from __future__ import annotations

import pandas as pd

from ml.config import DEFAULT_CONFIG, ForecastConfig


def _normalize_frequency(frequency: str) -> str:
    """Normalize frequency strings to lowercase to avoid pandas/Prophet issues."""
    return frequency.strip().lower()


def resample_hospital_timeseries(
    df: pd.DataFrame,
    hospital_id: str,
    config: ForecastConfig = DEFAULT_CONFIG,
    frequency: str | None = None,
    interpolate_target: bool = True,
) -> pd.DataFrame:
    """Resample one hospital's series to a fixed cadence.

    Notes
    -----
    - Occupancy is resampled by mean within each bucket.
    - Capacity is forward/backward filled if available because it is typically
      slower-moving operational metadata than occupancy.
    """
    freq = _normalize_frequency(frequency or config.resample_frequency)

    hospital_df = df[df[config.hospital_id_column] == hospital_id].copy()
    if hospital_df.empty:
        raise ValueError(f"No data found for hospital_id='{hospital_id}'")

    hospital_df = hospital_df.sort_values(config.timestamp_column)
    hospital_df = hospital_df.set_index(config.timestamp_column)

    columns_to_resample = [config.target_column]
    if config.capacity_column in hospital_df.columns:
        columns_to_resample.append(config.capacity_column)

    resampled = hospital_df[columns_to_resample].resample(freq).mean()

    # Occupancy imputation strategy for prototype-scale use.
    if interpolate_target:
        resampled[config.target_column] = (
            resampled[config.target_column]
            .interpolate(method="time")
            .ffill()
            .bfill()
        )
    else:
        resampled[config.target_column] = (
            resampled[config.target_column]
            .ffill()
            .bfill()
        )

    # Capacity is normally steadier than target; simple fill is usually enough.
    if config.capacity_column in resampled.columns:
        resampled[config.capacity_column] = (
            resampled[config.capacity_column]
            .ffill()
            .bfill()
        )

    resampled[config.hospital_id_column] = hospital_id
    return resampled.reset_index()


def resample_all_hospitals(
    df: pd.DataFrame,
    config: ForecastConfig = DEFAULT_CONFIG,
    frequency: str | None = None,
    interpolate_target: bool = True,
) -> pd.DataFrame:
    """Resample every hospital independently and combine the results."""
    hospital_ids = df[config.hospital_id_column].dropna().unique().tolist()
    parts: list[pd.DataFrame] = []

    for hospital_id in hospital_ids:
        part = resample_hospital_timeseries(
            df=df,
            hospital_id=str(hospital_id),
            config=config,
            frequency=frequency,
            interpolate_target=interpolate_target,
        )
        parts.append(part)

    if not parts:
        raise ValueError("No hospital data available for resampling")

    combined = pd.concat(parts, ignore_index=True)
    return combined.sort_values(
        [config.hospital_id_column, config.timestamp_column],
        kind="mergesort",
    ).reset_index(drop=True)