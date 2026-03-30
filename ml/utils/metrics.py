"""Common regression metrics used for model comparison."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd


def _align(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align two series by position after coercing to numeric arrays."""
    true_values = pd.Series(y_true).reset_index(drop=True)
    pred_values = pd.Series(y_pred).reset_index(drop=True)

    true_values = pd.to_numeric(true_values, errors="coerce")
    pred_values = pd.to_numeric(pred_values, errors="coerce")

    aligned = pd.DataFrame({"y_true": true_values, "y_pred": pred_values}).dropna()
    if aligned.empty:
        raise ValueError("No valid overlapping numeric values available for metric computation")

    return aligned["y_true"].to_numpy(dtype=float), aligned["y_pred"].to_numpy(dtype=float)


def mean_absolute_error(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Compute the average absolute difference between actuals and predictions."""
    aligned_true, aligned_pred = _align(y_true, y_pred)
    return float(np.mean(np.abs(aligned_true - aligned_pred)))


def root_mean_squared_error(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Compute RMSE, which penalizes larger errors more strongly."""
    aligned_true, aligned_pred = _align(y_true, y_pred)
    return float(math.sqrt(np.mean((aligned_true - aligned_pred) ** 2)))


def mean_absolute_percentage_error(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Compute MAPE while safely ignoring zero-valued denominators."""
    aligned_true, aligned_pred = _align(y_true, y_pred)
    non_zero_mask = aligned_true != 0
    if not np.any(non_zero_mask):
        return 0.0
    ratios = np.abs((aligned_true[non_zero_mask] - aligned_pred[non_zero_mask]) / aligned_true[non_zero_mask])
    return float(np.mean(ratios) * 100.0)


def symmetric_mean_absolute_percentage_error(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    """Compute SMAPE, which is often more stable than MAPE."""
    aligned_true, aligned_pred = _align(y_true, y_pred)
    denominator = np.abs(aligned_true) + np.abs(aligned_pred)
    valid_mask = denominator != 0
    if not np.any(valid_mask):
        return 0.0
    values = 2.0 * np.abs(aligned_pred[valid_mask] - aligned_true[valid_mask]) / denominator[valid_mask]
    return float(np.mean(values) * 100.0)


def summarize_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> Dict[str, float]:
    """Return the most useful metrics in one dictionary."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "smape": symmetric_mean_absolute_percentage_error(y_true, y_pred),
    }