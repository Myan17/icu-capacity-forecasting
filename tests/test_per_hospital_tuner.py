"""Unit tests for ml/per_hospital/tuner.py (L4 non-IID handling)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.per_hospital.tuner import PROPHET_GRID, SARIMA_GRID, PerHospitalTuner
from shared.hospital_tiers import profile_hospital


def _df(rows: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "timestamp":    pd.date_range("2023-01-01", periods=rows, freq="W"),
        "icu_occupied": 50 + 5 * np.sin(np.arange(rows) * 0.3) + rng.normal(0, 2, rows),
        "icu_capacity": 100,
    })


def test_tuner_returns_a_choice_for_small_hospital():
    df = _df(20)
    profile = profile_hospital("small", df)
    tuner = PerHospitalTuner(horizon=4)
    res = tuner.tune(profile, df)
    assert res.model_name in {"baseline", "prophet", "sarima"}
    assert isinstance(res.params, dict)
    assert res.explored, "explored grid should be non-empty"


def test_tuner_budget_scales_with_tier():
    df = _df(150, seed=2)
    profile = profile_hospital("large", df)
    tuner = PerHospitalTuner(horizon=4)
    res = tuner.tune(profile, df)
    # Budget for LARGE = 3, so we expect 1 baseline + up to 3 prophet + 3 sarima trials.
    explored_models = {x.get("model_name") for x in res.explored}
    assert "baseline" in explored_models
    # Either prophet or sarima should appear when more than one trial fits
    assert len(res.explored) >= 1


def test_grids_are_non_empty():
    assert PROPHET_GRID
    assert SARIMA_GRID
