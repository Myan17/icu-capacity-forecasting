"""Hospital tier classification for workload-aware scheduling.

Implements two related classifications used across the platform:

1. **Workload tier (L1: MOS-style)** — how heavy each pipeline is for this hospital.
   Drives Lambda memory sizing and Monte Carlo sample budgets.

2. **TiFL tier (L4: tiered federated learning)** — speed/availability class used to
   decide async vs. sync update cadence in the federated forecasting simulation.

Tiers are derived from the snapshot history (row count + occupancy variance) so
they self-calibrate as hospitals accumulate data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


class WorkloadTier(str, Enum):
    """L1: MOS-style hospital classification (small/medium/large workload)."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class TiFLTier(str, Enum):
    """L4: TiFL tier — fast/medium/slow update cadence for FedAT."""

    TIER_1 = "tier_1"  # Fast: large + low variance → sync, every cycle
    TIER_2 = "tier_2"  # Medium: moderate data → async, every 2 cycles
    TIER_3 = "tier_3"  # Slow: small / noisy → async, every 4 cycles


@dataclass(frozen=True)
class HospitalProfile:
    hospital_id: str
    rows: int
    mean_occupancy: float
    occupancy_variance: float
    workload_tier: WorkloadTier
    tifl_tier: TiFLTier

    @property
    def fl_weight(self) -> float:
        """Weight for federated weighted aggregation: prefer hospitals with more
        data and lower noise. Inverse-variance weighting clipped for stability."""
        var = max(self.occupancy_variance, 1e-3)
        return float(self.rows / var)

    @property
    def update_period(self) -> int:
        """How often (in scheduler cycles) this hospital should be re-forecast."""
        return {TiFLTier.TIER_1: 1, TiFLTier.TIER_2: 2, TiFLTier.TIER_3: 4}[self.tifl_tier]


_WORKLOAD_BOUNDS = {
    WorkloadTier.SMALL: (0, 30),     # < 30 rows
    WorkloadTier.MEDIUM: (30, 100),  # 30–99 rows
    WorkloadTier.LARGE: (100, 10**9),
}


def classify_workload(rows: int) -> WorkloadTier:
    """Map snapshot row count to a workload tier."""
    for tier, (lo, hi) in _WORKLOAD_BOUNDS.items():
        if lo <= rows < hi:
            return tier
    return WorkloadTier.SMALL


def classify_tifl(rows: int, variance: float) -> TiFLTier:
    """TiFL tier: combine data volume and noise.

    - Tier 1: large data and low variance → fast cadence (sync each cycle)
    - Tier 2: moderate data or moderate noise
    - Tier 3: small data or high variance → slow cadence
    """
    if rows >= 100 and variance < 50.0:
        return TiFLTier.TIER_1
    if rows >= 30:
        return TiFLTier.TIER_2
    return TiFLTier.TIER_3


def profile_hospital(
    hospital_id: str,
    history: Sequence[Mapping[str, float]] | pd.DataFrame,
    occupancy_col: str = "icu_occupied",
) -> HospitalProfile:
    """Build a HospitalProfile from raw snapshot history.

    Accepts either a DataFrame or a list of dict-like records (DynamoDB items).
    """
    if isinstance(history, pd.DataFrame):
        df = history
    else:
        df = pd.DataFrame(list(history))

    if df.empty or occupancy_col not in df.columns:
        return HospitalProfile(
            hospital_id=hospital_id,
            rows=0,
            mean_occupancy=0.0,
            occupancy_variance=0.0,
            workload_tier=WorkloadTier.SMALL,
            tifl_tier=TiFLTier.TIER_3,
        )

    series = pd.to_numeric(df[occupancy_col], errors="coerce").dropna().astype(float)
    rows = int(len(series))
    mean = float(series.mean()) if rows > 0 else 0.0
    var = float(series.var(ddof=0)) if rows > 1 else 0.0

    return HospitalProfile(
        hospital_id=hospital_id,
        rows=rows,
        mean_occupancy=mean,
        occupancy_variance=var,
        workload_tier=classify_workload(rows),
        tifl_tier=classify_tifl(rows, var),
    )


def profile_many(
    histories: Mapping[str, Sequence[Mapping[str, float]] | pd.DataFrame],
    occupancy_col: str = "icu_occupied",
) -> dict[str, HospitalProfile]:
    return {hid: profile_hospital(hid, hist, occupancy_col) for hid, hist in histories.items()}


def rank_by_priority(
    profiles: Iterable[HospitalProfile],
    risk_lookup: Mapping[str, float] | None = None,
) -> list[HospitalProfile]:
    """L2: order hospitals so high-risk ones are processed first.

    Priority key: current occupancy ratio (if provided) descending, then row count
    descending. Ties broken by hospital_id for determinism.
    """
    risk_lookup = risk_lookup or {}
    return sorted(
        profiles,
        key=lambda p: (
            -float(risk_lookup.get(p.hospital_id, 0.0)),
            -p.rows,
            p.hospital_id,
        ),
    )


# Lambda memory recommendations (MB) — used by the SAM template Mappings and by
# ad-hoc local benchmarks. Values mirror template.yaml::WorkloadTierMap.
LAMBDA_MEMORY_MB: dict[WorkloadTier, int] = {
    WorkloadTier.SMALL: 1024,
    WorkloadTier.MEDIUM: 2048,
    WorkloadTier.LARGE: 3008,
}

LAMBDA_TIMEOUT_SEC: dict[WorkloadTier, int] = {
    WorkloadTier.SMALL: 180,
    WorkloadTier.MEDIUM: 360,
    WorkloadTier.LARGE: 600,
}

# How many Monte Carlo CI samples to draw per tier — small hospitals don't need
# (and can't afford the latency of) 500 samples.
MC_SAMPLES_PER_TIER: dict[WorkloadTier, int] = {
    WorkloadTier.SMALL: 100,
    WorkloadTier.MEDIUM: 300,
    WorkloadTier.LARGE: 500,
}


def recommended_resources(profile: HospitalProfile) -> dict[str, int]:
    """Suggested Lambda config + MC budget for a given hospital profile."""
    return {
        "memory_mb": LAMBDA_MEMORY_MB[profile.workload_tier],
        "timeout_sec": LAMBDA_TIMEOUT_SEC[profile.workload_tier],
        "mc_samples": MC_SAMPLES_PER_TIER[profile.workload_tier],
    }


__all__ = [
    "WorkloadTier",
    "TiFLTier",
    "HospitalProfile",
    "classify_workload",
    "classify_tifl",
    "profile_hospital",
    "profile_many",
    "rank_by_priority",
    "recommended_resources",
    "LAMBDA_MEMORY_MB",
    "LAMBDA_TIMEOUT_SEC",
    "MC_SAMPLES_PER_TIER",
]
