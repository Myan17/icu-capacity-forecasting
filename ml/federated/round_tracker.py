"""RoundTracker — TiFL participation gating by tier (L4).

Spec: docs/superpowers/specs/2026-04-27-distributed-ml-features-design.md

Each FL round, hospitals participate only if their tier's update period
divides the current round number. This implements the *TiFL* tier-frequency
schedule:

    LARGE  : every round   (period 1)
    MEDIUM : every 2 rounds (period 2)
    SMALL  : every 4 rounds (period 4)

Round 0 is treated as the cold-start: every tier participates so the global
model has data from all clients before the staggered cadence kicks in.
"""
from __future__ import annotations

from ml.workload.profiler import Tier

_PARTICIPATION_FREQUENCY: dict[Tier, int] = {
    Tier.LARGE: 1,
    Tier.MEDIUM: 2,
    Tier.SMALL: 4,
}


class RoundTracker:
    """Track FL round number and gate participation by tier."""

    def __init__(self, current_round: int = 0) -> None:
        self.current_round = current_round

    def should_participate(self, tier: Tier) -> bool:
        freq = _PARTICIPATION_FREQUENCY[tier]
        return self.current_round % freq == 0

    def advance(self) -> None:
        self.current_round += 1


__all__ = ["RoundTracker"]
