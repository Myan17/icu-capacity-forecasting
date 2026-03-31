from __future__ import annotations

from ml.evaluation.metrics import compute_forecast_metrics


def test_compute_forecast_metrics() -> None:
    y_true = [10, 12, 14]
    y_pred = [11, 12, 13]

    metrics = compute_forecast_metrics(y_true, y_pred)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["mape"] >= 0