"""Unit tests for shared/hospital_tiers.py (L1 + L4 classifications)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.hospital_tiers import (
    LAMBDA_MEMORY_MB,
    MC_SAMPLES_PER_TIER,
    HospitalProfile,
    TiFLTier,
    WorkloadTier,
    classify_tifl,
    classify_workload,
    profile_hospital,
    profile_many,
    rank_by_priority,
    recommended_resources,
)


def _df(rows: int, mean: float = 50.0, var: float = 25.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "icu_occupied": rng.normal(mean, max(var, 1.0) ** 0.5, rows),
    })


def test_classify_workload_thresholds():
    assert classify_workload(0)   == WorkloadTier.SMALL
    assert classify_workload(29)  == WorkloadTier.SMALL
    assert classify_workload(30)  == WorkloadTier.MEDIUM
    assert classify_workload(99)  == WorkloadTier.MEDIUM
    assert classify_workload(100) == WorkloadTier.LARGE
    assert classify_workload(5000) == WorkloadTier.LARGE


@pytest.mark.parametrize("rows,var,expected", [
    (200, 10, TiFLTier.TIER_1),
    (150,  5, TiFLTier.TIER_1),
    (100, 100, TiFLTier.TIER_2),
    ( 50, 30, TiFLTier.TIER_2),
    ( 20, 10, TiFLTier.TIER_3),
    (  5, 100, TiFLTier.TIER_3),
])
def test_classify_tifl(rows, var, expected):
    assert classify_tifl(rows, var) == expected


def test_profile_hospital_empty():
    profile = profile_hospital("ghost", pd.DataFrame())
    assert profile.rows == 0
    assert profile.workload_tier == WorkloadTier.SMALL
    assert profile.tifl_tier == TiFLTier.TIER_3


def test_profile_hospital_normal():
    df = _df(200, mean=60.0, var=12.0, seed=1)
    profile = profile_hospital("hosp", df)
    assert profile.rows == 200
    assert 50 < profile.mean_occupancy < 70
    assert profile.workload_tier == WorkloadTier.LARGE


def test_recommended_resources_aligns_with_tier_tables():
    df = _df(150, var=5, seed=2)
    profile = profile_hospital("h", df)
    res = recommended_resources(profile)
    assert res["memory_mb"] == LAMBDA_MEMORY_MB[profile.workload_tier]
    assert res["mc_samples"] == MC_SAMPLES_PER_TIER[profile.workload_tier]


def test_fl_weight_is_inverse_variance_like():
    df_low = _df(100, var=1.0, seed=3)
    df_high = _df(100, var=100.0, seed=4)
    p_low  = profile_hospital("low",  df_low)
    p_high = profile_hospital("high", df_high)
    assert p_low.fl_weight > p_high.fl_weight


def test_update_period_matches_tier():
    p = HospitalProfile(
        hospital_id="t",
        rows=10,
        mean_occupancy=10.0,
        occupancy_variance=10.0,
        workload_tier=WorkloadTier.SMALL,
        tifl_tier=TiFLTier.TIER_3,
    )
    assert p.update_period == 4


def test_rank_by_priority_sorts_high_risk_first():
    profiles = [
        profile_hospital("a", _df(50, var=20, seed=10)),
        profile_hospital("b", _df(50, var=20, seed=11)),
        profile_hospital("c", _df(50, var=20, seed=12)),
    ]
    ranked = rank_by_priority(profiles, risk_lookup={"a": 0.95, "b": 0.30, "c": 0.10})
    assert [p.hospital_id for p in ranked] == ["a", "b", "c"]


def test_profile_many_works_on_dict_of_lists():
    histories = {
        "h1": [{"icu_occupied": 50}, {"icu_occupied": 55}, {"icu_occupied": 60}],
        "h2": pd.DataFrame({"icu_occupied": [10, 12, 11]}),
    }
    profiles = profile_many(histories)
    assert "h1" in profiles and "h2" in profiles
    assert profiles["h1"].rows == 3
