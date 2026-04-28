"""Federated learning simulation (Lecture 4: TiFL + FedAT).

Public API:
  * :class:`FederatedClient`       — per-hospital local trainer
  * :class:`FederatedAggregator`   — FedAvg / FedAvgM / weighted variants
  * :class:`FedATCoordinator`      — TiFL + FedAT scheduler

Run via ``python -m scripts.run_federated`` (see scripts/run_federated.py).
"""
from ml.federated.client import FederatedClient, ClientUpdate, ClientReport
from ml.federated.aggregator import (
    FederatedAggregator,
    GlobalModelState,
    fedavg,
    weighted_fedavg,
)
from ml.federated.fedat import FedATCoordinator, FederationRoundResult

__all__ = [
    "FederatedClient",
    "ClientUpdate",
    "ClientReport",
    "FederatedAggregator",
    "GlobalModelState",
    "fedavg",
    "weighted_fedavg",
    "FedATCoordinator",
    "FederationRoundResult",
]
