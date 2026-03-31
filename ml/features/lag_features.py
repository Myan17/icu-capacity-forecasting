from __future__ import annotations

import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lag_periods: list[int],
    rolling_window: int,
) -> pd.DataFrame:
    """
    Add lag and rolling mean features for a single-hospital time series.
    Assumes the dataframe is already sorted by time.
    """
    result = df.copy()

    for lag in lag_periods:
        result[f"{target_col}_lag_{lag}"] = result[target_col].shift(lag)

    result[f"{target_col}_roll_mean_{rolling_window}"] = (
        result[target_col].shift(1).rolling(rolling_window).mean()
    )
    result[f"{target_col}_diff_1"] = result[target_col].diff(1)
    return result