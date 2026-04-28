"""FederatedAggregator — FedAvg over per-hospital weight dicts (L4).

Spec: docs/superpowers/specs/2026-04-27-distributed-ml-features-design.md

This is the *spec-conformant* aggregator used by the forecast Lambda's FL
hook. It takes the list of per-hospital weight dicts produced by
:class:`ml.federated.local_trainer.LocalTrainer` and computes the FedAvg
global model:

    global_last_value = Σ (n_i / N_total) * local_last_value_i

It also picks the dominant model name by total training rows, returns the
client count + total rows for diagnostics, and (when configured with an S3
client + bucket) uploads the global model to ``fl/global_model.json``.

The numpy-θ aggregator used by the FedAT simulator now lives in
:mod:`ml.federated.parameter_aggregator` as ``ParameterAggregator``.
"""
from __future__ import annotations

import json
from typing import Any


class FederatedAggregator:
    """Compute FedAvg over per-hospital weight dicts and persist the global model."""

    def __init__(self, s3_client=None, data_bucket: str = "") -> None:
        self._s3 = s3_client
        self._bucket = data_bucket

    def aggregate(
        self,
        round_n: int,
        weight_dicts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not weight_dicts:
            return {}

        total_rows = sum(int(w["n_rows"]) for w in weight_dicts)
        if total_rows == 0:
            return {}

        global_last_value = sum(
            float(w["last_value"]) * (int(w["n_rows"]) / total_rows)
            for w in weight_dicts
        )

        model_type_rows: dict[str, int] = {}
        for w in weight_dicts:
            mn = w["model_name"]
            model_type_rows[mn] = model_type_rows.get(mn, 0) + int(w["n_rows"])
        dominant_model = max(model_type_rows, key=lambda k: model_type_rows[k])

        global_model: dict[str, Any] = {
            "round":             round_n,
            "num_clients":       len(weight_dicts),
            "global_last_value": global_last_value,
            "dominant_model":    dominant_model,
            "total_rows":        total_rows,
        }

        if self._bucket and self._s3:
            self._s3.put_object(
                Bucket=self._bucket,
                Key="fl/global_model.json",
                Body=json.dumps(global_model).encode(),
            )

        return global_model


__all__ = ["FederatedAggregator"]
