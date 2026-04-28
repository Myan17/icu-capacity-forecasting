"""LocalTrainer — per-hospital local model fit + weight extraction (L4).

Spec: docs/superpowers/specs/2026-04-27-distributed-ml-features-design.md

For each FL round, each hospital trains its own ``BestModelSelector`` on its
own data (no raw rows leave the client), then we extract a small dict of
"weights" — really model-specific summary parameters — that gets serialised
to JSON and uploaded to ``s3://{bucket}/fl/round_{N}/{hospital_id}_weights.json``.

The aggregator (``ml/federated/aggregator.py::FederatedAggregator``) then
performs FedAvg over those dicts.

Why a flat dict and not a tensor?  The forecasters in this project are a
mixed bag — Baseline (last value), SARIMA (AR/MA params), Prophet (changepoint
priors). A heterogeneous dict keeps the aggregator simple while still
exercising the spec's federated flow.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ml.config import DEFAULT_CONFIG
from ml.model_selection.selector import BestModelSelector


class LocalTrainer:
    """Train on a single hospital's data and extract serializable weights."""

    def __init__(self, s3_client=None, data_bucket: str = "") -> None:
        self._s3 = s3_client
        self._bucket = data_bucket

    def train_and_extract_weights(
        self,
        hospital_id: str,
        df: pd.DataFrame,
        round_n: int,
    ) -> dict[str, Any]:
        model_name, model = self._select_model(df)
        weights = self._extract_weights(hospital_id, df, model_name, model, round_n)
        if self._bucket and self._s3:
            key = f"fl/round_{round_n}/{hospital_id}_weights.json"
            self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=json.dumps(weights).encode(),
            )
        return weights

    def _select_model(self, df: pd.DataFrame) -> tuple[str, Any]:
        horizon = min(DEFAULT_CONFIG.test_size, max(len(df) // 4, 2))
        selector = BestModelSelector(
            metric=DEFAULT_CONFIG.selection_metric,
            horizon=horizon,
            timestamp_col="timestamp",
            target_col="icu_occupied",
            frequency="W",
        )
        result = selector.select_best_model(df.copy())
        return result.best_model_name, result.best_model

    def _extract_weights(
        self,
        hospital_id: str,
        df: pd.DataFrame,
        model_name: str,
        model: Any,
        round_n: int,
    ) -> dict[str, Any]:
        last_value = float(df["icu_occupied"].iloc[-1]) if len(df) > 0 else 0.0
        weights: dict[str, Any] = {
            "hospital_id": hospital_id,
            "model_name":  model_name,
            "n_rows":      len(df),
            "last_value":  last_value,
            "round":       round_n,
        }

        if model_name == "baseline" and hasattr(model, "last_value") and model.last_value is not None:
            weights["params"] = {"last_value": float(model.last_value)}
        elif model_name == "sarima" and hasattr(model, "fitted_model") and model.fitted_model is not None:
            try:
                weights["params"] = {
                    "ar_params": model.fitted_model.arparams.tolist(),
                    "ma_params": model.fitted_model.maparams.tolist(),
                }
            except Exception:
                weights["params"] = {"last_value": last_value}
        else:
            weights["params"] = {"last_value": last_value}

        return weights


__all__ = ["LocalTrainer"]
