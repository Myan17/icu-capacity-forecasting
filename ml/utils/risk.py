"""Risk labeling helpers for turning forecasts into operational alerts."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ml.config import DEFAULT_CONFIG, ForecastConfig


def assign_risk_label(
    predicted_occupied: float,
    capacity: Optional[float],
    config: ForecastConfig = DEFAULT_CONFIG,
) -> str:
    """Convert a predicted occupancy value into GREEN / YELLOW / RED.

    If capacity is unknown or invalid, the function defaults to GREEN so the
    pipeline remains operational. This policy can be tightened later.
    """
    if capacity is None:
        return "GREEN"
    if pd.isna(capacity) or capacity <= 0:
        return "GREEN"

    predicted_occupied = float(predicted_occupied)
    ratio = predicted_occupied / float(capacity)

    if ratio < config.green_threshold:
        return "GREEN"
    if ratio < config.yellow_threshold:
        return "YELLOW"
    return "RED"


def add_risk_column(
    forecast_df: pd.DataFrame,
    capacity: Optional[float],
    prediction_column: str = "predicted_icu_occupied",
    config: ForecastConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Return a copy of the forecast frame with risk-related columns."""
    if prediction_column not in forecast_df.columns:
        raise ValueError(f"Prediction column '{prediction_column}' not found in forecast frame")

    enriched = forecast_df.copy()
    enriched[prediction_column] = pd.to_numeric(enriched[prediction_column], errors="coerce")

    enriched["capacity_used_for_risk"] = capacity
    enriched["occupancy_ratio"] = (
        enriched[prediction_column] / capacity if capacity is not None and capacity > 0 else None
    )
    enriched["risk_level"] = enriched[prediction_column].apply(
        lambda value: assign_risk_label(value, capacity, config)
    )
    return enriched