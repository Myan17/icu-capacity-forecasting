from __future__ import annotations

import pandas as pd


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise an error if required columns are missing."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_no_empty_group(df: pd.DataFrame, group_col: str) -> None:
    """Ensure grouping column has no missing values."""
    if df[group_col].isna().any():
        raise ValueError(f"Grouping column '{group_col}' contains missing values")


def validate_weekly_frequency(df: pd.DataFrame, timestamp_col: str) -> None:
    """
    Validate that timestamps are approximately weekly for a single hospital time series.
    This is a warning-style validator: it raises only when the series is clearly invalid.
    """
    series = df[timestamp_col].sort_values().dropna()
    if len(series) < 2:
        return

    diffs = series.diff().dropna().dt.days
    allowed = diffs.isin([7])

    # Allow a small number of irregular gaps, but reject heavily inconsistent data.
    if allowed.mean() < 0.8:
        raise ValueError(
            "Dataset does not appear to have weekly frequency. "
            f"Observed day gaps: {sorted(diffs.unique().tolist())}"
        )


def validate_target_not_all_missing(df: pd.DataFrame, target_col: str) -> None:
    """Ensure target column has at least some usable data."""
    if df[target_col].dropna().empty:
        raise ValueError(f"Target column '{target_col}' is entirely missing")