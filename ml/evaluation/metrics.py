from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_forecast_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute standard regression metrics for forecasting."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    denominator = np.clip(np.abs(y_true), 1e-8, None)
    mape = np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
    }