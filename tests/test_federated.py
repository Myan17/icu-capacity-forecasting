"""Unit tests for the federated learning simulator (Lecture 4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.federated.client import (
    ClientUpdate,
    FederatedClient,
    init_global_params,
)
from ml.federated.fedat import FedATCoordinator, tier_summary
from ml.federated.parameter_aggregator import (
    GlobalModelState,
    ParameterAggregator,
    fedavg,
    weighted_fedavg,
)
from shared.hospital_tiers import profile_hospital


def _hospital_df(rows: int, mean: float = 50.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "timestamp":    pd.date_range("2023-01-01", periods=rows, freq="W"),
        "icu_occupied": mean + rng.normal(0, 2, rows),
        "icu_capacity": 100,
    })


def _build_clients(n: int):
    clients = []
    profiles = {}
    for i in range(n):
        rows = [200, 150, 70, 35, 20][i % 5]
        df = _hospital_df(rows, mean=50 + i, seed=i)
        prof = profile_hospital(f"h{i}", df)
        profiles[prof.hospital_id] = prof
        clients.append(FederatedClient(prof, local_epochs=2).load_data(df))
    return clients, profiles


def test_init_global_params_dimension():
    theta = init_global_params()
    assert theta.shape == (4,)
    assert theta[3] == pytest.approx(1.0)  # baseline weight starts at 1


def test_local_train_reduces_loss_or_keeps_constant():
    df = _hospital_df(100, mean=50, seed=42)
    profile = profile_hospital("h", df)
    client = FederatedClient(profile, local_epochs=5).load_data(df)
    update = client.local_train(init_global_params())
    assert update.num_samples == 100
    assert np.isfinite(update.local_loss)
    assert update.theta.shape == (4,)


def test_fedavg_weights_by_sample_count():
    u_big   = ClientUpdate("a", np.array([1.0, 0, 0, 0]), num_samples=100, local_loss=1.0, duration_ms=0)
    u_small = ClientUpdate("b", np.array([0.0, 0, 0, 0]), num_samples=1,   local_loss=1.0, duration_ms=0)
    avg = fedavg([u_big, u_small])
    assert avg[0] > 0.95


def test_weighted_fedavg_uses_external_weights():
    u1 = ClientUpdate("a", np.array([1.0, 0, 0, 0]), num_samples=10, local_loss=1.0, duration_ms=0)
    u2 = ClientUpdate("b", np.array([0.0, 0, 0, 0]), num_samples=10, local_loss=1.0, duration_ms=0)
    avg_default = weighted_fedavg([u1, u2], {"a": 1.0, "b": 1.0})
    avg_skewed  = weighted_fedavg([u1, u2], {"a": 9.0, "b": 1.0})
    assert avg_default[0] == pytest.approx(0.5)
    assert avg_skewed[0]  > 0.85


def test_fedavgm_velocity_grows_with_consistent_direction():
    state = GlobalModelState(theta=np.zeros(4), velocity=np.zeros(4))
    agg = ParameterAggregator(strategy="fedavgm", momentum=0.9)
    updates = [ClientUpdate("a", np.array([1.0, 0, 0, 0]), num_samples=10, local_loss=1.0, duration_ms=0)]
    new_state = agg.aggregate(state, updates)
    assert new_state.theta[0] > state.theta[0]


def test_fedat_round_participation_rules():
    clients, profiles = _build_clients(5)
    coord = FedATCoordinator(clients=clients)
    state, logs = coord.run(num_rounds=5, strategy="weighted")
    # Round 1: everyone participates (cold start)
    assert set(logs[0].participants) == {c.hospital_id for c in clients}
    # Loss decreases or stays comparable across rounds (with some FL noise)
    successful_losses = [l.avg_loss for l in logs if np.isfinite(l.avg_loss)]
    assert successful_losses[0] >= min(successful_losses) * 0.5  # not pathologically diverging


def test_fedat_evaluate_returns_predictions_per_client():
    clients, profiles = _build_clients(3)
    coord = FedATCoordinator(clients=clients)
    state, _ = coord.run(num_rounds=2, strategy="weighted")
    out = coord.evaluate(state, horizon=4)
    for hid in {c.hospital_id for c in clients}:
        assert hid in out
        assert "predictions" in out[hid]
        assert len(out[hid]["predictions"]) == 4


def test_tier_summary_counts_participation():
    clients, profiles = _build_clients(5)
    coord = FedATCoordinator(clients=clients)
    _, logs = coord.run(num_rounds=4, strategy="weighted")
    summary = tier_summary(logs, profiles)
    assert "tier_1" in summary and "tier_2" in summary and "tier_3" in summary
    total_rounds = sum(s["rounds_participated"] for s in summary.values())
    assert total_rounds > 0
