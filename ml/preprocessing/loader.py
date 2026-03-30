"""Functions for loading raw hospital occupancy data from CSV or Parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.config import DEFAULT_CONFIG, ForecastConfig


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names so downstream code is less fragile."""
    normalized = df.copy()
    normalized.columns = [str(col).strip().lower() for col in normalized.columns]
    return normalized


def _validate_columns(df: pd.DataFrame, config: ForecastConfig) -> None:
    """Ensure the input table contains the required schema."""
    required = set(config.required_columns())
    missing = required.difference(df.columns)
    if missing:
        missing_display = ", ".join(sorted(missing))
        raise ValueError(f"Input file is missing required columns: {missing_display}")


def _coerce_basic_types(df: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    """Parse timestamps and numeric fields into expected types."""
    coerced = df.copy()

    # Parse timestamps first so sort order is meaningful.
    coerced[config.timestamp_column] = pd.to_datetime(
        coerced[config.timestamp_column],
        errors="coerce",
    )

    # Convert numeric columns if present.
    coerced[config.target_column] = pd.to_numeric(
        coerced[config.target_column],
        errors="coerce",
    )

    if config.capacity_column in coerced.columns:
        coerced[config.capacity_column] = pd.to_numeric(
            coerced[config.capacity_column],
            errors="coerce",
        )

    return coerced


def _sort_rows(df: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    """Sort rows by hospital and timestamp."""
    return df.sort_values(
        [config.hospital_id_column, config.timestamp_column],
        kind="mergesort",
    ).reset_index(drop=True)


def load_csv(path: str | Path, config: ForecastConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Load a CSV file and apply basic schema normalization."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)
    _validate_columns(df, config)
    df = _coerce_basic_types(df, config)
    return _sort_rows(df, config)


def load_parquet(path: str | Path, config: ForecastConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Load a Parquet file and apply basic schema normalization."""
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Input file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    df = _normalize_columns(df)
    _validate_columns(df, config)
    df = _coerce_basic_types(df, config)
    return _sort_rows(df, config)


def load_dataset(path: str | Path, config: ForecastConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Load a dataset from a supported file type."""
    input_path = Path(path)
    suffix = input_path.suffix.lower()

    if suffix not in config.supported_input_formats:
        supported = ", ".join(sorted(config.supported_input_formats))
        raise ValueError(
            f"Unsupported input format '{suffix}'. Supported formats: {supported}"
        )

    if suffix == ".csv":
        return load_csv(input_path, config)

    if suffix == ".parquet":
        return load_parquet(input_path, config)

    # This line is defensive; the earlier check should catch unsupported types.
    raise ValueError(f"Unhandled input format: {suffix}")