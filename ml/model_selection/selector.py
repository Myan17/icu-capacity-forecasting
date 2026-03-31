"""Select the best forecasting model for a hospital time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import pandas as pd

from ml.models.baseline import BaselineForecaster
from ml.models.prophet_model import ProphetForecaster
from ml.models.sarima_model import SarimaForecaster


@dataclass
class ModelSelectionResult:
    best_model_name: str
    best_model: Any
    best_score: float
    metric_used: str
    all_results: List[Dict[str, Any]]
    train_rows: int
    validation_rows: int


class BestModelSelector:
    """Train candidate forecasters, compare them on validation data, and pick the best one."""

    def __init__(
        self,
        metric: str = "rmse",
        horizon: int = 8,
        timestamp_col: str = "timestamp",
        target_col: str = "icu_occupied",
        frequency: str = "W",
) -> None:
        self.metric = metric.lower()
        self.horizon = horizon
        self.timestamp_col = timestamp_col
        self.target_col = target_col
        self.frequency = frequency

        self.candidate_models = {
            "baseline": lambda: BaselineForecaster(target_col=self.target_col),
            "prophet": lambda: ProphetForecaster(
                target_col=self.target_col,
                frequency=self.frequency,
            ),
            "sarima": lambda: SarimaForecaster(target_col=self.target_col),
        }

    def select_best_model(self, df: pd.DataFrame) -> ModelSelectionResult:
        """
        Expects df with at least:
          - ds : datetime column
          - y  : target column (icu occupancy)
        """
        df = df.sort_values(self.timestamp_col).reset_index(drop=True)

        if len(df) <= self.horizon:
            raise ValueError(
                f"Not enough rows to do model selection. "
                f"Need more than horizon={self.horizon}, got {len(df)} rows."
            )

        train_df = df.iloc[:-self.horizon].copy()
        valid_df = df.iloc[-self.horizon:].copy()

        all_results: List[Dict[str, Any]] = []
        best_model_name = None
        best_model = None
        best_score = float("inf")

        for model_name, model_factory in self.candidate_models.items():
            try:
                model = model_factory()
                self._fit_model(model, train_df)

                preds = model.predict(self.horizon)

                # Normalize predictions
                pred_values = self._extract_predictions(preds)
                actual_values = valid_df[self.target_col].to_numpy()

                # Match lengths safely
                min_len = min(len(pred_values), len(actual_values))
                pred_values = pred_values[:min_len]
                actual_values = actual_values[:min_len]

                metrics = self._compute_metrics(actual_values, pred_values)
                score = metrics[self.metric]

                result = {
                    "model_name": model_name,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "mape": metrics["mape"],
                    "actuals": actual_values.tolist(),
                    "predictions": pred_values.tolist(),
                    "status": "success",
                }
                all_results.append(result)

                if score < best_score:
                    best_score = score
                    best_model_name = model_name
                    best_model = model

            except Exception as e:
                print(f"[selector] model '{model_name}' failed: {e}")
                all_results.append(
                    {
                        "model_name": model_name,
                        "status": "failed",
                        "error": str(e),
                    }
                )

        if best_model_name is None or best_model is None:
            raise RuntimeError("All candidate models failed during selection.")

        # Retrain best model on full history before final use
        final_model = self.candidate_models[best_model_name]()
        self._fit_model(final_model, df)

        return ModelSelectionResult(
            best_model_name=best_model_name,
            best_model=final_model,
            best_score=best_score,
            metric_used=self.metric,
            all_results=all_results,
            train_rows=len(train_df),
            validation_rows=len(valid_df),
        )

    def _extract_predictions(self, preds: Any) -> np.ndarray:
        """
        Convert model-specific prediction output into a flat numpy array.
        Adjust this if your model outputs differ.
        """
        if isinstance(preds, pd.DataFrame):
            # Prophet often returns yhat
            if "yhat" in preds.columns:
                return preds["yhat"].to_numpy()
            # fallback: first numeric column
            numeric_cols = preds.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return preds[numeric_cols[0]].to_numpy()

        if isinstance(preds, pd.Series):
            return preds.to_numpy()

        if isinstance(preds, list):
            return np.array(preds, dtype=float)

        if isinstance(preds, np.ndarray):
            return preds.astype(float)

        raise ValueError(f"Unsupported prediction output type: {type(preds)}")

    def _compute_metrics(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> Dict[str, float]:
        error = actual - predicted

        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error ** 2)))

        # Avoid divide-by-zero in MAPE
        non_zero_mask = actual != 0
        if np.any(non_zero_mask):
            mape = float(
                np.mean(
                    np.abs(
                        (actual[non_zero_mask] - predicted[non_zero_mask])
                        / actual[non_zero_mask]
                    )
                )
                * 100
            )
        else:
            mape = float("inf")

        return {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
        }
    
    def _fit_model(self, model, train_df: pd.DataFrame) -> None:
        if isinstance(model, ProphetForecaster):
            model.fit(train_df, timestamp_col=self.timestamp_col)
        else:
            model.fit(train_df)