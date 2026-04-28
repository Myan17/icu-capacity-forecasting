"""FL unit tests (spec: docs/superpowers/specs/2026-04-27).

Combines the three FL components into a single test module:

  * RoundTracker        — TiFL participation gating per tier.
  * LocalTrainer        — per-hospital model fit + weight-dict export + S3 upload.
  * FederatedAggregator — dict-based FedAvg + S3 global-model upload.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from ml.federated.aggregator import FederatedAggregator
from ml.federated.local_trainer import LocalTrainer
from ml.federated.round_tracker import RoundTracker
from ml.workload.profiler import Tier


# ── RoundTracker ──────────────────────────────────────────────────────────────

def test_large_participates_every_round():
    tracker = RoundTracker(current_round=0)
    assert tracker.should_participate(Tier.LARGE) is True
    tracker.advance()
    assert tracker.should_participate(Tier.LARGE) is True


def test_medium_participates_every_2_rounds():
    tracker = RoundTracker(current_round=0)
    assert tracker.should_participate(Tier.MEDIUM) is True
    tracker.advance()
    assert tracker.should_participate(Tier.MEDIUM) is False
    tracker.advance()
    assert tracker.should_participate(Tier.MEDIUM) is True


def test_small_participates_every_4_rounds():
    tracker = RoundTracker(current_round=0)
    assert tracker.should_participate(Tier.SMALL) is True
    for _ in range(3):
        tracker.advance()
    assert tracker.should_participate(Tier.SMALL) is False
    tracker.advance()  # round 4
    assert tracker.should_participate(Tier.SMALL) is True


def test_advance_increments_round():
    tracker = RoundTracker(current_round=2)
    tracker.advance()
    assert tracker.current_round == 3


def test_round_zero_all_tiers_participate():
    tracker = RoundTracker(current_round=0)
    for tier in Tier:
        assert tracker.should_participate(tier) is True


# ── LocalTrainer ──────────────────────────────────────────────────────────────

def _make_hospital_df(rows: int = 20) -> pd.DataFrame:
    np.random.seed(0)
    ts = pd.date_range("2023-01-01", periods=rows, freq="W")
    return pd.DataFrame({
        "timestamp":    ts,
        "icu_occupied": np.random.normal(50, 8, size=rows),
        "icu_capacity": [100] * rows,
        "hospital_id":  ["H_test"] * rows,
    })


def test_local_trainer_returns_weight_dict():
    df = _make_hospital_df(20)
    trainer = LocalTrainer(s3_client=None, data_bucket="")
    weights = trainer.train_and_extract_weights("H_test", df, round_n=0)
    for key in ("hospital_id", "model_name", "n_rows", "last_value", "round"):
        assert key in weights
    assert weights["hospital_id"] == "H_test"
    assert weights["n_rows"] == 20
    assert weights["round"] == 0


def test_local_trainer_last_value_matches_df_tail():
    df = _make_hospital_df(20)
    trainer = LocalTrainer(s3_client=None, data_bucket="")
    weights = trainer.train_and_extract_weights("H_test", df, round_n=1)
    expected_last = float(df["icu_occupied"].iloc[-1])
    assert abs(weights["last_value"] - expected_last) < 1e-6


def test_local_trainer_uploads_to_s3_when_bucket_set():
    df = _make_hospital_df(20)
    mock_s3 = MagicMock()
    trainer = LocalTrainer(s3_client=mock_s3, data_bucket="test-bucket")
    trainer.train_and_extract_weights("H_test", df, round_n=2)
    mock_s3.put_object.assert_called_once()
    call_kwargs = mock_s3.put_object.call_args[1]
    assert call_kwargs["Bucket"] == "test-bucket"
    assert "fl/round_2/H_test_weights.json" in call_kwargs["Key"]


def test_local_trainer_skips_upload_when_no_bucket():
    df = _make_hospital_df(20)
    mock_s3 = MagicMock()
    trainer = LocalTrainer(s3_client=mock_s3, data_bucket="")
    trainer.train_and_extract_weights("H_test", df, round_n=0)
    mock_s3.put_object.assert_not_called()


# ── FederatedAggregator ───────────────────────────────────────────────────────

def _weight(
    hospital_id: str, last_value: float, n_rows: int,
    model_name: str = "baseline",
) -> dict:
    return {
        "hospital_id": hospital_id,
        "model_name":  model_name,
        "n_rows":      n_rows,
        "last_value":  last_value,
        "round":       0,
        "params":      {"last_value": last_value},
    }


def test_fedavg_is_weighted_mean_by_rows():
    weights = [
        _weight("H1", last_value=60.0, n_rows=40),
        _weight("H2", last_value=40.0, n_rows=10),
    ]
    result = FederatedAggregator().aggregate(round_n=0, weight_dicts=weights)
    # FedAvg: (60*40 + 40*10) / 50 = (2400+400)/50 = 56.0
    assert abs(result["global_last_value"] - 56.0) < 1e-6


def test_aggregate_returns_required_keys():
    weights = [_weight("H1", 50.0, 20), _weight("H2", 60.0, 30)]
    result = FederatedAggregator().aggregate(round_n=1, weight_dicts=weights)
    for key in ("round", "num_clients", "global_last_value", "dominant_model", "total_rows"):
        assert key in result


def test_aggregate_num_clients_correct():
    weights = [_weight(f"H{i}", 50.0, 10) for i in range(5)]
    result = FederatedAggregator().aggregate(round_n=0, weight_dicts=weights)
    assert result["num_clients"] == 5


def test_aggregate_dominant_model_is_most_common():
    weights = [
        _weight("H1", 50.0, 30, model_name="sarima"),
        _weight("H2", 50.0, 20, model_name="sarima"),
        _weight("H3", 50.0, 10, model_name="baseline"),
    ]
    result = FederatedAggregator().aggregate(round_n=0, weight_dicts=weights)
    assert result["dominant_model"] == "sarima"


def test_aggregate_empty_list_returns_empty():
    result = FederatedAggregator().aggregate(round_n=0, weight_dicts=[])
    assert result == {}


def test_aggregate_uploads_global_model_to_s3():
    mock_s3 = MagicMock()
    weights = [_weight("H1", 50.0, 20), _weight("H2", 60.0, 30)]
    FederatedAggregator(s3_client=mock_s3, data_bucket="test-bucket").aggregate(
        round_n=3, weight_dicts=weights,
    )
    mock_s3.put_object.assert_called_once()
    call_kwargs = mock_s3.put_object.call_args[1]
    assert call_kwargs["Key"] == "fl/global_model.json"
    body = json.loads(call_kwargs["Body"])
    assert body["round"] == 3
