"""FedAT — Federated Learning with Asynchronous Tiers (Lecture 4).

The TiFL paper buckets clients by tier (fast/medium/slow) so each round only
samples from a single tier; FedAT extends that with **async** updates: fast
tiers contribute every round, slow tiers contribute every k-th round, and
their contributions are blended into the global model with a staleness
penalty.

Implementation choices for the simulator:

  * Tier 1 → participates every round (sync within tier)
  * Tier 2 → participates every 2 rounds
  * Tier 3 → participates every 4 rounds (async, oldest contribution)
  * Staleness penalty: weight *= 1 / (1 + α * staleness_in_rounds)

The coordinator orchestrates the full round-robin, returning a per-round
log that can be fed to MLflow or printed for the report.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

import numpy as np

from ml.federated.client import ClientUpdate, FederatedClient, init_global_params
from ml.federated.parameter_aggregator import GlobalModelState, ParameterAggregator
from shared.hospital_tiers import HospitalProfile, TiFLTier


@dataclass
class FederationRoundResult:
    round_num: int
    participants: List[str]
    skipped: List[str]
    aggregator_strategy: str
    avg_loss: float
    duration_ms: float
    staleness: Mapping[str, int]
    weights_used: Mapping[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round":               self.round_num,
            "participants":        list(self.participants),
            "skipped":             list(self.skipped),
            "aggregator_strategy": self.aggregator_strategy,
            "avg_loss":            round(self.avg_loss, 4),
            "duration_ms":         round(self.duration_ms, 2),
            "staleness":           dict(self.staleness),
            "weights_used":        {k: round(v, 4) for k, v in self.weights_used.items()},
        }


class FedATCoordinator:
    """Orchestrates TiFL + FedAT over a fleet of FederatedClients.

    Parameters
    ----------
    staleness_alpha : float
        Penalty coefficient on per-round staleness (α=0 → no penalty).
    fairness_floor : float
        Lower bound on each client's effective weight; prevents starvation
        of small / Tier 3 hospitals.
    """

    def __init__(
        self,
        clients: List[FederatedClient],
        staleness_alpha: float = 0.5,
        fairness_floor: float = 0.05,
    ) -> None:
        if not clients:
            raise ValueError("FedAT requires at least one client.")
        self.clients = {c.hospital_id: c for c in clients}
        self.staleness_alpha = staleness_alpha
        self.fairness_floor = fairness_floor
        self.last_seen: Dict[str, ClientUpdate] = {}
        self.last_round_seen: Dict[str, int] = {}

    # ── tier participation rules ─────────────────────────────────────────────

    def _participates_this_round(self, profile: HospitalProfile, round_num: int) -> bool:
        """Round 1 always includes everyone (cold-start). After that, each
        tier participates every ``update_period`` rounds (FedAT staggering)."""
        period = profile.update_period
        if period <= 1 or round_num == 1:
            return True
        return ((round_num - 1) % period) == 0

    def _staleness_for(self, hospital_id: str, round_num: int) -> int:
        last = self.last_round_seen.get(hospital_id)
        if last is None:
            return round_num
        return max(round_num - last, 0)

    def _fl_weight(self, profile: HospitalProfile, staleness: int) -> float:
        base = profile.fl_weight
        decayed = base / (1.0 + self.staleness_alpha * staleness)
        return max(decayed, self.fairness_floor)

    # ── public API ──────────────────────────────────────────────────────────

    def run(
        self,
        num_rounds: int = 5,
        strategy: str = "weighted",
    ) -> tuple[GlobalModelState, List[FederationRoundResult]]:
        state = GlobalModelState(theta=init_global_params())
        round_logs: List[FederationRoundResult] = []

        for r in range(1, num_rounds + 1):
            t0 = time.perf_counter()
            participants: List[ClientUpdate] = []
            skipped: List[str] = []
            weights_used: Dict[str, float] = {}
            staleness_map: Dict[str, int] = {}

            for hid, client in self.clients.items():
                if self._participates_this_round(client.profile, r):
                    update = client.local_train(state.theta)
                    self.last_seen[hid] = update
                    self.last_round_seen[hid] = r
                    participants.append(update)
                    staleness_map[hid] = 0
                else:
                    if hid in self.last_seen:
                        # FedAT: include cached stale update with penalty
                        cached = self.last_seen[hid]
                        stale_rounds = self._staleness_for(hid, r)
                        participants.append(cached)
                        staleness_map[hid] = stale_rounds
                    else:
                        skipped.append(hid)

            for u in participants:
                profile = self.clients[u.hospital_id].profile
                weights_used[u.hospital_id] = self._fl_weight(profile, staleness_map.get(u.hospital_id, 0))

            if not participants:
                # Nothing to aggregate this round (no fresh + no cached); roll over.
                round_logs.append(FederationRoundResult(
                    round_num=r,
                    participants=[],
                    skipped=skipped,
                    aggregator_strategy=strategy,
                    avg_loss=float("nan"),
                    duration_ms=(time.perf_counter() - t0) * 1000.0,
                    staleness=dict(staleness_map),
                    weights_used={},
                ))
                state = GlobalModelState(
                    theta=state.theta,
                    round_num=state.round_num + 1,
                    history=state.history,
                    velocity=state.velocity,
                )
                continue

            aggregator = ParameterAggregator(
                strategy=strategy,
                external_weights=weights_used if strategy == "weighted" else None,
            )
            state = aggregator.aggregate(state, participants)

            avg_loss = float(np.mean([u.local_loss for u in participants if np.isfinite(u.local_loss)] or [float("nan")]))
            round_logs.append(FederationRoundResult(
                round_num=r,
                participants=[u.hospital_id for u in participants],
                skipped=skipped,
                aggregator_strategy=strategy,
                avg_loss=avg_loss,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                staleness=dict(staleness_map),
                weights_used=dict(weights_used),
            ))

        return state, round_logs

    def evaluate(self, state: GlobalModelState, horizon: int = 8) -> Dict[str, Dict[str, Any]]:
        """Return per-client forecasts under the aggregated global θ."""
        out: Dict[str, Dict[str, Any]] = {}
        for hid, client in self.clients.items():
            preds = client.predict_with_global(state.theta, horizon=horizon)
            out[hid] = {
                "tier":        client.profile.tifl_tier.value,
                "predictions": preds.tolist(),
            }
        return out


def tier_summary(round_logs: List[FederationRoundResult],
                 profiles: Mapping[str, HospitalProfile]) -> Dict[str, Dict[str, float]]:
    """Aggregate participation/loss stats per TiFL tier across the run."""
    tiers: Dict[str, Dict[str, float]] = {
        TiFLTier.TIER_1.value: {"rounds": 0, "losses": []},
        TiFLTier.TIER_2.value: {"rounds": 0, "losses": []},
        TiFLTier.TIER_3.value: {"rounds": 0, "losses": []},
    }
    for log in round_logs:
        for hid in log.participants:
            if hid not in profiles:
                continue
            tier = profiles[hid].tifl_tier.value
            tiers[tier]["rounds"] += 1
            tiers[tier]["losses"].append(log.avg_loss)
    summary = {}
    for tier, data in tiers.items():
        losses = data["losses"]
        summary[tier] = {
            "rounds_participated": data["rounds"],
            "mean_loss": float(np.mean(losses)) if losses else float("nan"),
        }
    return summary


__all__ = ["FedATCoordinator", "FederationRoundResult", "tier_summary"]
