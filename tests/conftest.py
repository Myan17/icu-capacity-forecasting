"""Shared pytest fixtures for hospital-forecasting test suite."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _good_df(**overrides) -> pd.DataFrame:
    """Return a minimal valid snapshot DataFrame."""
    data = {
        "hospital_id":  ["H001", "H001", "H002"],
        "timestamp":    pd.to_datetime(["2024-01-07", "2024-01-14", "2024-01-07"]),
        "icu_capacity": [100, 100, 50],
        "icu_occupied": [80, 85, 40],
        "admissions":   [5, 6, 3],
        "discharges":   [4, 5, 2],
        "transfers":    [1, 1, 0],
        "ed_arrivals":  [10, 12, 8],
        "staffing_level": [0.9, 0.88, 0.95],
    }
    data.update(overrides)
    return pd.DataFrame(data)


@pytest.fixture
def good_df() -> pd.DataFrame:
    return _good_df()


@pytest.fixture
def missing_col_df() -> pd.DataFrame:
    df = _good_df()
    return df.drop(columns=["icu_capacity"])


@pytest.fixture
def null_timestamp_df() -> pd.DataFrame:
    df = _good_df()
    df.loc[0, "timestamp"] = None
    return df


@pytest.fixture
def negative_occupied_df() -> pd.DataFrame:
    df = _good_df()
    df.loc[0, "icu_occupied"] = -5
    return df


@pytest.fixture
def zero_capacity_df() -> pd.DataFrame:
    df = _good_df()
    df.loc[0, "icu_capacity"] = 0
    return df


@pytest.fixture
def implausible_occupancy_df() -> pd.DataFrame:
    df = _good_df()
    df.loc[0, "icu_occupied"] = 200  # > 1.5 × 100
    return df
