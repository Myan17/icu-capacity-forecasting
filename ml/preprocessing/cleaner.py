"""Data cleaning helpers for ICU forecasting inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from ml.config import DEFAULT_CONFIG, ForecastConfig


@dataclass
class CleaningSummary:
    """Compact summary of what happened during cleaning."""

    input_rows: int
    output_rows: int
    dropped_missing_timestamp: int
    dropped_missing_target: int
    dropped_missing_hospital_id: int
    dropped_invalid_capacity: int
    dropped_negative_target: int
    dropped_target_exceeds_capacity: int
    duplicate_rows_removed: int

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly version of the summary."""
        return {
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "dropped_missing_timestamp": self.dropped_missing_timestamp,
            "dropped_missing_target": self.dropped_missing_target,
            "dropped_missing_hospital_id": self.dropped_missing_hospital_id,
            "dropped_invalid_capacity": self.dropped_invalid_capacity,
            "dropped_negative_target": self.dropped_negative_target,
            "dropped_target_exceeds_capacity": self.dropped_target_exceeds_capacity,
            "duplicate_rows_removed": self.duplicate_rows_removed,
        }


def clean_timeseries(
    df: pd.DataFrame,
    config: ForecastConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Clean the dataset and return only the cleaned DataFrame."""
    cleaned, _ = clean_timeseries_with_summary(df, config)
    return cleaned


def clean_timeseries_with_summary(
    df: pd.DataFrame,
    config: ForecastConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, CleaningSummary]:
    """Clean the dataset and return both cleaned data and a summary."""
    cleaned = df.copy()
    input_rows = len(cleaned)

    # Drop rows where hospital_id is missing.
    missing_hospital_mask = cleaned[config.hospital_id_column].isna()
    dropped_missing_hospital_id = int(missing_hospital_mask.sum())
    cleaned = cleaned.loc[~missing_hospital_mask].copy()

    # Drop rows where timestamp parsing failed upstream.
    missing_ts_mask = cleaned[config.timestamp_column].isna()
    dropped_missing_timestamp = int(missing_ts_mask.sum())
    cleaned = cleaned.loc[~missing_ts_mask].copy()

    # Ensure target is numeric.
    cleaned[config.target_column] = pd.to_numeric(
        cleaned[config.target_column],
        errors="coerce",
    )
    missing_target_mask = cleaned[config.target_column].isna()
    dropped_missing_target = int(missing_target_mask.sum())
    cleaned = cleaned.loc[~missing_target_mask].copy()

    # Capacity is optional for some flows, but if present it should be valid.
    dropped_invalid_capacity = 0
    dropped_target_exceeds_capacity = 0
    if config.capacity_column in cleaned.columns:
        cleaned[config.capacity_column] = pd.to_numeric(
            cleaned[config.capacity_column],
            errors="coerce",
        )

        invalid_capacity_mask = cleaned[config.capacity_column].notna() & (
            cleaned[config.capacity_column] <= 0
        )
        dropped_invalid_capacity += int(invalid_capacity_mask.sum())
        cleaned = cleaned.loc[~invalid_capacity_mask].copy()

        if config.drop_rows_with_missing_capacity:
            missing_capacity_mask = cleaned[config.capacity_column].isna()
            dropped_invalid_capacity += int(missing_capacity_mask.sum())
            cleaned = cleaned.loc[~missing_capacity_mask].copy()

        if config.enforce_capacity_upper_bound:
            comparable_mask = (
                cleaned[config.capacity_column].notna()
                & cleaned[config.target_column].notna()
            )
            exceeds_capacity_mask = comparable_mask & (
                cleaned[config.target_column] > cleaned[config.capacity_column]
            )
            dropped_target_exceeds_capacity = int(exceeds_capacity_mask.sum())
            cleaned = cleaned.loc[~exceeds_capacity_mask].copy()

    # Domain rule: occupancy cannot be negative.
    negative_target_mask = cleaned[config.target_column] < 0
    dropped_negative_target = int(negative_target_mask.sum())
    cleaned = cleaned.loc[~negative_target_mask].copy()

    # Keep the latest row for duplicate (hospital_id, timestamp).
    before_dedup = len(cleaned)
    cleaned = cleaned.drop_duplicates(
        subset=[config.hospital_id_column, config.timestamp_column],
        keep="last",
    )
    duplicate_rows_removed = before_dedup - len(cleaned)

    cleaned = cleaned.sort_values(
        [config.hospital_id_column, config.timestamp_column],
        kind="mergesort",
    ).reset_index(drop=True)

    summary = CleaningSummary(
        input_rows=input_rows,
        output_rows=len(cleaned),
        dropped_missing_timestamp=dropped_missing_timestamp,
        dropped_missing_target=dropped_missing_target,
        dropped_missing_hospital_id=dropped_missing_hospital_id,
        dropped_invalid_capacity=dropped_invalid_capacity,
        dropped_negative_target=dropped_negative_target,
        dropped_target_exceeds_capacity=dropped_target_exceeds_capacity,
        duplicate_rows_removed=duplicate_rows_removed,
    )

    return cleaned, summary