"""WorkloadProfiler — classify hospitals into compute tiers (L1: MOS).

Spec: docs/superpowers/specs/2026-04-27-distributed-ml-features-design.md

Tier rules (rows = ``len(df)``, variance = ``df["icu_occupied"].var()``):
    LARGE  : rows >= 30 AND variance > 50          (priority 1, mc=1000)
    MEDIUM : rows >= 15 OR  variance >= 20         (priority 2, mc=500)
    SMALL  : everything else                       (priority 3, mc=200)

``priority_score = rows * 0.6 + variance_normalized * 0.4`` where
``variance_normalized = min(variance / 200.0, 1.0)``.

The profiler caches results per ``hospital_id`` so multiple lookups within a
warm Lambda container don't re-classify the same series.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Tier(str, Enum):
    LARGE = "LARGE"
    MEDIUM = "MEDIUM"
    SMALL = "SMALL"


@dataclass
class WorkloadProfile:
    hospital_id: str
    tier: Tier
    priority_score: float
    mc_samples: int
    reason: str


_VARIANCE_MAX = 200.0  # normalisation cap for priority score
_TIER_MC = {Tier.LARGE: 1000, Tier.MEDIUM: 500, Tier.SMALL: 200}


class WorkloadProfiler:
    """Classify hospitals into compute tiers based on row count + variance."""

    def __init__(self) -> None:
        self._cache: dict[str, WorkloadProfile] = {}

    def profile(self, hospital_id: str, df: pd.DataFrame) -> WorkloadProfile:
        if hospital_id in self._cache:
            return self._cache[hospital_id]

        rows = len(df)
        variance = (
            float(df["icu_occupied"].var())
            if rows > 1 and "icu_occupied" in df.columns
            else 0.0
        )
        variance_normalized = min(variance / _VARIANCE_MAX, 1.0)
        priority_score = rows * 0.6 + variance_normalized * 0.4

        if rows >= 30 and variance > 50:
            tier = Tier.LARGE
        elif rows >= 15 or variance >= 20:
            tier = Tier.MEDIUM
        else:
            tier = Tier.SMALL

        result = WorkloadProfile(
            hospital_id=hospital_id,
            tier=tier,
            priority_score=priority_score,
            mc_samples=_TIER_MC[tier],
            reason=f"rows={rows}, variance={variance:.1f}",
        )
        self._cache[hospital_id] = result
        return result

    def clear_cache(self) -> None:
        self._cache.clear()


__all__ = ["Tier", "WorkloadProfile", "WorkloadProfiler"]
