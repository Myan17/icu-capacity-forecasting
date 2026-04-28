"""TierScheduler unit tests (spec: docs/superpowers/specs/2026-04-27).

Asserts:
  * Tier ordering — LARGE before MEDIUM before SMALL.
  * SMALL bin-packing — at most 5 hospitals per batch.
  * Staleness ordering within a tier when ``last_forecast_ages`` is provided.
  * MEDIUM / LARGE never get bin-packed (one hospital per batch).
  * JobBatch fields populated correctly.
"""
from __future__ import annotations

from ml.scheduler.priority_queue import JobBatch, TierScheduler
from ml.workload.profiler import Tier, WorkloadProfile


def _profile(hospital_id: str, tier: Tier, score: float = 10.0) -> WorkloadProfile:
    mc = {Tier.LARGE: 1000, Tier.MEDIUM: 500, Tier.SMALL: 200}[tier]
    return WorkloadProfile(
        hospital_id=hospital_id, tier=tier,
        priority_score=score, mc_samples=mc, reason="test",
    )


def test_large_before_small():
    profiles = [_profile("S1", Tier.SMALL), _profile("L1", Tier.LARGE)]
    batches = TierScheduler().schedule(profiles)
    assert batches[0].tier == Tier.LARGE


def test_large_before_medium_before_small():
    profiles = [
        _profile("S1", Tier.SMALL),
        _profile("L1", Tier.LARGE),
        _profile("M1", Tier.MEDIUM),
    ]
    tiers = [b.tier for b in TierScheduler().schedule(profiles)]
    assert tiers.index(Tier.LARGE) < tiers.index(Tier.MEDIUM)
    assert tiers.index(Tier.MEDIUM) < tiers.index(Tier.SMALL)


def test_small_hospitals_bin_packed_max_5():
    profiles = [_profile(f"S{i}", Tier.SMALL) for i in range(7)]
    batches = TierScheduler().schedule(profiles)
    small_batches = [b for b in batches if b.tier == Tier.SMALL]
    assert len(small_batches) == 2
    assert len(small_batches[0].hospital_ids) == 5
    assert len(small_batches[1].hospital_ids) == 2


def test_no_batch_exceeds_5():
    profiles = [_profile(f"S{i}", Tier.SMALL) for i in range(12)]
    batches = TierScheduler().schedule(profiles)
    for b in batches:
        assert len(b.hospital_ids) <= 5


def test_medium_not_binpacked():
    profiles = [_profile(f"M{i}", Tier.MEDIUM) for i in range(3)]
    batches = TierScheduler().schedule(profiles)
    assert len(batches) == 3
    for b in batches:
        assert len(b.hospital_ids) == 1


def test_staleness_orders_within_tier():
    profiles = [_profile("L1", Tier.LARGE), _profile("L2", Tier.LARGE)]
    ages = {"L1": 10.0, "L2": 50.0}
    batches = TierScheduler().schedule(profiles, last_forecast_ages=ages)
    large_ids = [b.hospital_ids[0] for b in batches if b.tier == Tier.LARGE]
    assert large_ids[0] == "L2"  # most stale first


def test_empty_input_returns_empty():
    assert TierScheduler().schedule([]) == []


def test_job_batch_fields_populated():
    profiles = [_profile("L1", Tier.LARGE)]
    batch = TierScheduler().schedule(profiles)[0]
    assert isinstance(batch, JobBatch)
    assert batch.priority == 1
    assert batch.estimated_duration_s > 0
    assert batch.hospital_ids == ["L1"]
