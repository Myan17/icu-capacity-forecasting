from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_dataset(
    path: str | Path,
    timestamp_col: str,
    required_columns: list[str],
) -> pd.DataFrame:
    """Load cleaned CSV dataset and validate basic schema."""
    df = pd.read_csv(path, low_memory=False, dtype={"hospital_id": str})

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    if df[timestamp_col].isna().any():
        invalid_count = int(df[timestamp_col].isna().sum())
        raise ValueError(
            f"Found {invalid_count} invalid timestamps in column '{timestamp_col}'"
        )

    return df