"""Parameter-vector federated aggregation strategies.

This module hosts the *numpy θ-vector* aggregator used by the Holt-Winters-lite
``FederatedClient`` and ``FedATCoordinator`` simulators (Lecture 4 stretch).

It is intentionally separate from :mod:`ml.federated.aggregator` (which hosts
the spec-conformant *weight-dict* ``FederatedAggregator``) so the two
abstractions don't fight over the same class name.

Three strategies are implemented:

  * ``fedavg``           — McMahan et al. 2017: average client params weighted
                            by sample count.
  * ``weighted_fedavg``  — apply a custom per-client weight (e.g. inverse
                            variance, fairness, FedAT staleness).
  * ``fedavg_with_momentum`` (FedAvgM) — keep a running velocity term across
                            rounds; useful for noisy / heterogeneous fleets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from ml.federated.client import ClientUpdate


@dataclass
class GlobalModelState:
    theta: np.ndarray
    round_num: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    velocity: np.ndarray | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round":   self.round_num,
            "theta":   self.theta.tolist(),
            "history": self.history,
        }


def _stack(updates: Iterable[ClientUpdate]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(updates)
    if not rows:
        raise ValueError("No client updates supplied to aggregator.")
    thetas = np.stack([u.theta for u in rows])
    n_samples = np.array([max(u.num_samples, 1) for u in rows], dtype=float)
    losses = np.array([u.local_loss for u in rows], dtype=float)
    return thetas, n_samples, losses


def fedavg(updates: Iterable[ClientUpdate]) -> np.ndarray:
    """Vanilla FedAvg: weighted average by sample count."""
    thetas, n_samples, _ = _stack(updates)
    weights = n_samples / n_samples.sum()
    return (thetas * weights[:, None]).sum(axis=0)


def weighted_fedavg(
    updates: Iterable[ClientUpdate],
    weights: Mapping[str, float],
) -> np.ndarray:
    """Aggregate using *external* per-client weights (e.g. fairness boosts,
    FedAT staleness penalties)."""
    rows = list(updates)
    if not rows:
        raise ValueError("No client updates supplied.")
    thetas = np.stack([u.theta for u in rows])
    raw = np.array([max(weights.get(u.hospital_id, 1.0), 1e-9) for u in rows], dtype=float)
    raw /= raw.sum()
    return (thetas * raw[:, None]).sum(axis=0)


def fedavg_with_momentum(
    updates: Iterable[ClientUpdate],
    state: GlobalModelState,
    momentum: float = 0.9,
    lr: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """FedAvgM (Hsu et al. 2019). Returns (theta_new, velocity_new)."""
    thetas, n_samples, _ = _stack(updates)
    weights = n_samples / n_samples.sum()
    delta = (thetas * weights[:, None]).sum(axis=0) - state.theta
    velocity = state.velocity if state.velocity is not None else np.zeros_like(state.theta)
    velocity_new = momentum * velocity + delta
    theta_new = state.theta + lr * velocity_new
    return theta_new, velocity_new


class ParameterAggregator:
    """Stateful aggregator over numpy parameter vectors (multi-round + diagnostics)."""

    def __init__(
        self,
        strategy: str = "fedavg",
        external_weights: Mapping[str, float] | None = None,
        momentum: float = 0.9,
    ) -> None:
        self.strategy = strategy
        self.external_weights = external_weights
        self.momentum = momentum

    def aggregate(
        self,
        state: GlobalModelState,
        updates: List[ClientUpdate],
    ) -> GlobalModelState:
        if self.strategy == "fedavg":
            theta_new = fedavg(updates)
            velocity_new = state.velocity
        elif self.strategy == "weighted":
            if not self.external_weights:
                raise ValueError("weighted strategy requires external_weights")
            theta_new = weighted_fedavg(updates, self.external_weights)
            velocity_new = state.velocity
        elif self.strategy == "fedavgm":
            theta_new, velocity_new = fedavg_with_momentum(
                updates, state, momentum=self.momentum
            )
        else:
            raise ValueError(f"unknown FL strategy: {self.strategy}")

        avg_loss = float(np.mean([u.local_loss for u in updates if np.isfinite(u.local_loss)]))
        history_entry = {
            "round":     state.round_num + 1,
            "strategy":  self.strategy,
            "n_clients": len(updates),
            "avg_loss":  round(avg_loss, 4),
            "delta_norm": float(np.linalg.norm(theta_new - state.theta)),
            "participants": [u.hospital_id for u in updates],
        }
        return GlobalModelState(
            theta=theta_new,
            round_num=state.round_num + 1,
            history=state.history + [history_entry],
            velocity=velocity_new,
        )


__all__ = [
    "GlobalModelState",
    "fedavg",
    "weighted_fedavg",
    "fedavg_with_momentum",
    "ParameterAggregator",
]
