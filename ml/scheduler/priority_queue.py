"""TierScheduler — priority sort + SMALL bin-packing (L2: Borg / Kubernetes).

Spec: docs/superpowers/specs/2026-04-27-distributed-ml-features-design.md

Replaces the flat ``for hid in hospital_ids`` loop with a Borg-style scheduler
that:
  1. Sorts hospitals by ``(tier ASC, last_forecast_age DESC)`` — LARGE & stale
     hospitals run first.
  2. Bin-packs SMALL hospitals into batches of up to 5 (one batch shares one
     thread in the executor).
  3. Emits one ``JobBatch`` per LARGE / MEDIUM hospital and one per SMALL bin.

Each batch carries an estimated runtime so the executor can flag stragglers
later if it wants to.
"""
from __future__ import annotations

from dataclasses import dataclass

from ml.workload.profiler import Tier, WorkloadProfile

_SMALL_BIN_SIZE = 5
_PRIORITY = {Tier.LARGE: 1, Tier.MEDIUM: 2, Tier.SMALL: 3}
_ESTIMATED_DURATION_S = {Tier.LARGE: 120.0, Tier.MEDIUM: 60.0, Tier.SMALL: 20.0}


@dataclass
class JobBatch:
    hospital_ids: list[str]
    tier: Tier
    priority: int
    estimated_duration_s: float


class TierScheduler:
    """Sort hospitals by tier + staleness; bin-pack SMALL hospitals (≤5/batch)."""

    def schedule(
        self,
        profiles: list[WorkloadProfile],
        last_forecast_ages: dict[str, float] | None = None,
    ) -> list[JobBatch]:
        ages = last_forecast_ages or {}
        sorted_profiles = sorted(
            profiles,
            key=lambda p: (_PRIORITY[p.tier], -ages.get(p.hospital_id, 0.0)),
        )

        batches: list[JobBatch] = []
        small_bin: list[str] = []

        for profile in sorted_profiles:
            if profile.tier == Tier.SMALL:
                small_bin.append(profile.hospital_id)
                if len(small_bin) >= _SMALL_BIN_SIZE:
                    batches.append(_small_batch(small_bin[:]))
                    small_bin.clear()
            else:
                batches.append(JobBatch(
                    hospital_ids=[profile.hospital_id],
                    tier=profile.tier,
                    priority=_PRIORITY[profile.tier],
                    estimated_duration_s=_ESTIMATED_DURATION_S[profile.tier],
                ))

        if small_bin:
            batches.append(_small_batch(small_bin))

        return batches


def _small_batch(hospital_ids: list[str]) -> JobBatch:
    return JobBatch(
        hospital_ids=hospital_ids,
        tier=Tier.SMALL,
        priority=_PRIORITY[Tier.SMALL],
        estimated_duration_s=_ESTIMATED_DURATION_S[Tier.SMALL] * len(hospital_ids),
    )


__all__ = ["JobBatch", "TierScheduler"]
