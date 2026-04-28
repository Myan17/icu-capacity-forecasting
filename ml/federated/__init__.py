"""Federated learning simulation (Lecture 4: FedAvg / TiFL / FedAT).

Spec-conformant entry points (used by the forecast Lambda's FL hook):

  * :class:`FederatedAggregator`   — dict-based FedAvg over weight dicts
  * :class:`LocalTrainer`          — per-hospital training + S3 upload
  * :class:`RoundTracker`          — TiFL participation gating

Stretch / parameter-vector flavour (used by ``scripts/run_federated.py`` and
the FedAT simulator):

  * :class:`FederatedClient`       — per-hospital local trainer
  * :class:`ParameterAggregator`   — FedAvg / FedAvgM / weighted on numpy θ
  * :class:`FedATCoordinator`      — staggered async tier coordinator
"""
from ml.federated.aggregator import FederatedAggregator
from ml.federated.client import ClientReport, ClientUpdate, FederatedClient
from ml.federated.fedat import FedATCoordinator, FederationRoundResult
from ml.federated.local_trainer import LocalTrainer
from ml.federated.parameter_aggregator import (
    GlobalModelState,
    ParameterAggregator,
    fedavg,
    weighted_fedavg,
)
from ml.federated.round_tracker import RoundTracker

__all__ = [
    "FederatedAggregator",
    "FederatedClient",
    "ClientUpdate",
    "ClientReport",
    "LocalTrainer",
    "RoundTracker",
    "FedATCoordinator",
    "FederationRoundResult",
    "ParameterAggregator",
    "GlobalModelState",
    "fedavg",
    "weighted_fedavg",
]
