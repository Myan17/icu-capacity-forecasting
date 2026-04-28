"""Unit tests for shared/priority_queue.py (Lecture 2 / Borg)."""
from __future__ import annotations

import pandas as pd

from shared.hospital_tiers import profile_hospital
from shared.priority_queue import (
    ExecutionPool,
    HospitalSignal,
    PriorityScheduler,
    compute_priority,
    estimated_makespan_seconds,
    lpt_optimal_ratio,
    schedule_and_pack,
    utilisation_balance,
)


def _profile(hid: str, rows: int = 80, var: float = 10.0):
    df = pd.DataFrame({"icu_occupied": [50.0 + (i % 5) for i in range(rows)]})
    return profile_hospital(hid, df)


def test_compute_priority_increases_with_risk_signals():
    p = _profile("h")
    base, _ = compute_priority(p, occupancy_ratio=0.1, breach_prob=0.0)
    high, reason = compute_priority(p, occupancy_ratio=0.95, breach_prob=0.8)
    assert high > base
    assert "RED_OCC" in reason
    assert "HIGH_BP" in reason


def test_priority_scheduler_drains_in_score_order():
    p1 = _profile("low")
    p2 = _profile("mid")
    p3 = _profile("high")

    sched = PriorityScheduler()
    sched.submit_many([
        HospitalSignal(p1, occupancy_ratio=0.1),
        HospitalSignal(p2, occupancy_ratio=0.5),
        HospitalSignal(p3, occupancy_ratio=0.95, last_breach_prob=0.8),
    ])

    ordered = list(sched.drain())
    assert [it.hospital_id for it in ordered] == ["high", "mid", "low"]
    assert len(sched) == 0


def test_execution_pool_balances_load():
    profiles = [_profile(f"h{i}") for i in range(8)]
    signals = [
        HospitalSignal(p, occupancy_ratio=0.4 + (i * 0.05)) for i, p in enumerate(profiles)
    ]
    ordered, shards = schedule_and_pack(signals, num_shards=3)
    assert len(shards) == 3
    assert sum(len(s.items) for s in shards) == 8
    # LPT bound for 3 shards = 4/3 - 1/9 ≈ 1.22; balance ≥ 0.65 is realistic
    # for a discrete 8-item / 3-shard pack.
    assert utilisation_balance(shards) > 0.65
    assert estimated_makespan_seconds(shards, per_hospital_seconds=1.0) > 0


def test_execution_pool_lpt_bound():
    profiles = [_profile(f"h{i}") for i in range(10)]
    signals = [HospitalSignal(p, occupancy_ratio=0.2 + i * 0.05) for i, p in enumerate(profiles)]
    sched = PriorityScheduler()
    items = sched.submit_many(signals)
    pool = ExecutionPool(num_shards=3)
    shards = pool.pack(items)
    # LPT bound for m machines is at most 4/3 - 1/(3m) of OPT
    assert lpt_optimal_ratio(shards) <= 4 / 3 + 1e-9
    assert lpt_optimal_ratio(shards) >= 1.0  # always ≥ 1 (worst case = OPT)


def test_priority_scheduler_handles_empty():
    sched = PriorityScheduler()
    assert list(sched.drain()) == []
    assert utilisation_balance([]) == 1.0
    assert estimated_makespan_seconds([], 30.0) == 0.0
