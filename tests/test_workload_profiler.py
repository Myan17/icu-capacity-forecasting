"""WorkloadProfiler unit tests (spec: docs/superpowers/specs/2026-04-27).

Covers tier classification, MC sample assignment, priority score ordering,
and the warm-cache contract used by the forecast Lambda.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.workload.profiler import Tier, WorkloadProfile, WorkloadProfiler


def _make_df(rows: int, variance_target: float) -> pd.DataFrame:
    """Deterministic synthetic series with sample variance == ``variance_target``.

    We draw a small normal sample then standardize + rescale so the pandas
    sample variance (ddof=1) exactly matches ``variance_target``. This avoids
    the borderline-randomness problem you'd hit with raw ``np.random.normal``
    at small N — which would occasionally place us on the wrong side of the
    20/50 tier threshold.
    """
    rng = np.random.default_rng(42)
    raw = rng.standard_normal(size=rows)
    if rows > 1 and raw.std(ddof=1) > 0:
        raw = (raw - raw.mean()) / raw.std(ddof=1)
        raw = raw * (variance_target ** 0.5) + 50.0
    else:
        raw = raw + 50.0
    return pd.DataFrame({"icu_occupied": raw, "icu_capacity": [100] * rows})


def test_large_tier_assigned():
    df = _make_df(35, 80.0)
    profile = WorkloadProfiler().profile("H001", df)
    assert profile.tier == Tier.LARGE
    assert profile.mc_samples == 1000


def test_medium_tier_by_rows():
    df = _make_df(20, 5.0)
    profile = WorkloadProfiler().profile("H002", df)
    assert profile.tier == Tier.MEDIUM
    assert profile.mc_samples == 500


def test_medium_tier_by_variance():
    df = _make_df(10, 30.0)
    profile = WorkloadProfiler().profile("H003", df)
    assert profile.tier == Tier.MEDIUM


def test_small_tier_assigned():
    df = _make_df(8, 5.0)
    profile = WorkloadProfiler().profile("H004", df)
    assert profile.tier == Tier.SMALL
    assert profile.mc_samples == 200


def test_returns_workload_profile_type():
    df = _make_df(10, 5.0)
    profile = WorkloadProfiler().profile("H005", df)
    assert isinstance(profile, WorkloadProfile)
    assert profile.hospital_id == "H005"
    assert profile.reason != ""


def test_cache_returns_same_object():
    df = _make_df(35, 80.0)
    profiler = WorkloadProfiler()
    p1 = profiler.profile("H006", df)
    p2 = profiler.profile("H006", df)
    assert p1 is p2


def test_large_has_higher_priority_score_than_small():
    large_df = _make_df(35, 80.0)
    small_df = _make_df(8, 5.0)
    profiler = WorkloadProfiler()
    large_p = profiler.profile("H007", large_df)
    small_p = profiler.profile("H008", small_df)
    assert large_p.priority_score > small_p.priority_score


def test_clear_cache():
    df = _make_df(35, 80.0)
    profiler = WorkloadProfiler()
    p1 = profiler.profile("H009", df)
    profiler.clear_cache()
    p2 = profiler.profile("H009", df)
    assert p1 is not p2
