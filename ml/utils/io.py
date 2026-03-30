"""Helpers for writing and reading ML artifacts to disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def ensure_directory(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def write_json(payload: Dict[str, Any], output_path: str | Path) -> None:
    """Write a JSON payload with readable indentation."""
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, default=str)


def read_json(input_path: str | Path) -> Dict[str, Any]:
    """Read a JSON file into a Python dictionary."""
    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_dataframe_csv(df: pd.DataFrame, output_path: str | Path, index: bool = False) -> None:
    """Write a DataFrame to CSV."""
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    df.to_csv(output_path, index=index)


def write_dataframe_parquet(df: pd.DataFrame, output_path: str | Path, index: bool = False) -> None:
    """Write a DataFrame to Parquet."""
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    df.to_parquet(output_path, index=index)


def write_metrics_json(
    hospital_id: str,
    model_name: str,
    metrics: Dict[str, float],
    output_path: str | Path,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """Write model evaluation metrics to JSON."""
    payload = {
        "hospital_id": hospital_id,
        "model_name": model_name,
        "metrics": metrics,
        "metadata": metadata or {},
    }
    write_json(payload, output_path)