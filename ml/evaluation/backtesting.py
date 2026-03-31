from __future__ import annotations

import pandas as pd


def time_split(df: pd.DataFrame, test_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a single time series into train and test sets chronologically."""
    if len(df) <= test_size:
        raise ValueError(
            f"Not enough rows ({len(df)}) for test_size={test_size}. "
            "Need more history."
        )

    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    return train_df, test_df