"""Priority-based scheduler for batch forecasting (Lecture 2: Borg / Kubernetes).

Borg/K8s schedule pods by priority class so the most urgent work runs first
even under contention. We mirror that for hospital forecasting:

  1. Each hospital gets a *priority score* derived from current occupancy ratio,
     recent breach probability, and TiFL tier urgency.
  2. The scheduler returns hospitals in priority-descending order.
  3. The forecast Lambda processes them in that order so a 600-second budget
     is spent on the most likely-to-breach hospitals first.

The scheduler is pure-Python and has no AWS dependencies; the forecast handler
calls it after loading current-state data. A separate ``ExecutionPool`` packs
hospitals into bin-packed shards (analogous to Borg bin-packing) to drive
batch forecasting (see L2 / batch forecasting feature).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

from shared.hospital_tiers import HospitalProfile, TiFLTier


@dataclass(order=True, frozen=True)
class PriorityItem:
    """Heap-ordered priority entry. Lower numerical priority = processed first.

    We negate the natural priority score so that high-risk hospitals
    sort *before* low-risk ones in a min-heap.
    """

    sort_key: float
    hospital_id: str = field(compare=False)
    score: float = field(compare=False)
    reason: str = field(compare=False, default="")


def _tier_urgency(tier: TiFLTier) -> float:
    """Tier 1 hospitals have the most data → most reliable forecasts → highest
    operational urgency. Tier 3 still gets processed but later."""
    return {TiFLTier.TIER_1: 1.0, TiFLTier.TIER_2: 0.6, TiFLTier.TIER_3: 0.3}[tier]


def compute_priority(
    profile: HospitalProfile,
    occupancy_ratio: float = 0.0,
    breach_prob: float = 0.0,
    capacity_change_pct: float = 0.0,
) -> tuple[float, str]:
    """Combine signals into a single priority score in [0, 5].

    Components (weighted sum):
      * occupancy_ratio    × 2.0   — current load
      * breach_prob        × 2.0   — predicted breach risk
      * capacity_change    × 1.0   — sudden capacity shock
      * tier_urgency       × 0.5   — TiFL tier weight
    """
    score = (
        2.0 * occupancy_ratio
        + 2.0 * breach_prob
        + 1.0 * abs(capacity_change_pct)
        + 0.5 * _tier_urgency(profile.tifl_tier)
    )
    reasons = []
    if occupancy_ratio >= 0.9:
        reasons.append("RED_OCC")
    elif occupancy_ratio >= 0.75:
        reasons.append("YELLOW_OCC")
    if breach_prob >= 0.6:
        reasons.append("HIGH_BP")
    if abs(capacity_change_pct) >= 0.2:
        reasons.append("CAPACITY_SHIFT")
    reasons.append(profile.tifl_tier.value.upper())
    return float(score), ",".join(reasons)


@dataclass
class HospitalSignal:
    """Lightweight bundle of current-state signals for priority computation."""

    profile: HospitalProfile
    occupancy_ratio: float = 0.0
    last_breach_prob: float = 0.0
    capacity_change_pct: float = 0.0


class PriorityScheduler:
    """Borg-style priority scheduler.

    Add signals via ``submit`` (or in bulk via ``submit_many``), then iterate
    ``drain`` to pop hospitals in priority order. A second pass with
    ``ExecutionPool`` packs them into N parallel shards.
    """

    def __init__(self) -> None:
        self._heap: list[PriorityItem] = []

    def submit(self, signal: HospitalSignal) -> PriorityItem:
        score, reason = compute_priority(
            profile=signal.profile,
            occupancy_ratio=signal.occupancy_ratio,
            breach_prob=signal.last_breach_prob,
            capacity_change_pct=signal.capacity_change_pct,
        )
        item = PriorityItem(
            sort_key=-score,
            hospital_id=signal.profile.hospital_id,
            score=score,
            reason=reason,
        )
        heapq.heappush(self._heap, item)
        return item

    def submit_many(self, signals: Iterable[HospitalSignal]) -> list[PriorityItem]:
        return [self.submit(s) for s in signals]

    def drain(self) -> Iterator[PriorityItem]:
        """Yield items in priority-descending order, draining the heap."""
        while self._heap:
            yield heapq.heappop(self._heap)

    def snapshot(self) -> list[PriorityItem]:
        """Return a (sorted, non-destructive) view of the queue."""
        return sorted(self._heap, key=lambda it: it.sort_key)

    def __len__(self) -> int:
        return len(self._heap)


@dataclass
class Shard:
    """A bin of hospitals to run together in one parallel worker / Lambda."""

    index: int
    items: list[PriorityItem] = field(default_factory=list)
    total_score: float = 0.0

    def add(self, item: PriorityItem) -> None:
        self.items.append(item)
        self.total_score += item.score

    @property
    def hospital_ids(self) -> list[str]:
        return [it.hospital_id for it in self.items]


class ExecutionPool:
    """Bin-packs prioritised hospitals into ``num_shards`` parallel groups.

    This is the L2 'request grouping + execution pooling' feature: instead of
    one Lambda per hospital, we pack them so each shard's *load* is balanced.
    Greedy least-loaded-first packing (a.k.a. Longest Processing Time, LPT)
    is the simplest near-optimal scheduler for this multiprocessor job-shop.
    """

    def __init__(self, num_shards: int = 4) -> None:
        if num_shards < 1:
            raise ValueError("num_shards must be ≥ 1")
        self.num_shards = num_shards

    def pack(self, items: Sequence[PriorityItem]) -> list[Shard]:
        ordered = sorted(items, key=lambda it: -it.score)
        shards = [Shard(index=i) for i in range(self.num_shards)]
        for item in ordered:
            target = min(shards, key=lambda s: s.total_score)
            target.add(item)
        return shards


def schedule_and_pack(
    signals: Iterable[HospitalSignal],
    num_shards: int = 4,
) -> tuple[list[PriorityItem], list[Shard]]:
    """Convenience: prioritise signals and pack into shards in one call."""
    sched = PriorityScheduler()
    sched.submit_many(signals)
    ordered = list(sched.drain())
    shards = ExecutionPool(num_shards=num_shards).pack(ordered)
    return ordered, shards


def utilisation_balance(shards: Sequence[Shard]) -> float:
    """Borg-style balance metric: 1.0 = perfectly balanced, 0 = pathological.

    Computed as 1 - (max_load - min_load) / max_load when max_load > 0."""
    if not shards:
        return 1.0
    loads = [s.total_score for s in shards]
    hi, lo = max(loads), min(loads)
    if hi <= 0:
        return 1.0
    return float(1.0 - (hi - lo) / hi)


def estimated_makespan_seconds(
    shards: Sequence[Shard], per_hospital_seconds: float = 30.0
) -> float:
    """Worst-case wall-clock if shards run in parallel."""
    if not shards:
        return 0.0
    return max(len(s.items) * per_hospital_seconds for s in shards)


def lpt_optimal_ratio(shards: Sequence[Shard]) -> float:
    """Theoretical bound: LPT achieves at most (4/3 - 1/(3m)) × OPT.

    Returns the bound for the current shard count; used in the auto-optimizer
    to decide when packing further is no longer worthwhile."""
    m = max(len(shards), 1)
    return 4.0 / 3.0 - 1.0 / (3.0 * m)


__all__ = [
    "PriorityItem",
    "HospitalSignal",
    "PriorityScheduler",
    "Shard",
    "ExecutionPool",
    "compute_priority",
    "schedule_and_pack",
    "utilisation_balance",
    "estimated_makespan_seconds",
    "lpt_optimal_ratio",
]
