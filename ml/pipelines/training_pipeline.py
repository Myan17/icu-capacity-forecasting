from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import matplotlib.pyplot as plt

from ml.config import DEFAULT_CONFIG, ForecastConfig
from ml.evaluation.backtesting import time_split
from ml.evaluation.metrics import compute_forecast_metrics
from ml.features.lag_features import add_lag_features
from ml.models.baseline import BaselineForecaster
from ml.models.prophet_model import ProphetForecaster
from ml.models.sarima_model import SarimaForecaster
from ml.model_selection.selector import BestModelSelector
from ml.preprocessing.loader import load_dataset
from ml.preprocessing.validator import (
    validate_no_empty_group,
    validate_required_columns,
    validate_target_not_all_missing,
    validate_weekly_frequency,
)


@dataclass
class TrainingResult:
    """Container describing the pipeline output."""
    hospital_id: str
    model_name: str
    fitted_model: Any
    training_rows: int
    artifact_path: str | None
    metadata: dict[str, Any]


class TrainingPipeline:
    """End-to-end weekly forecasting pipeline for one hospital."""

    def __init__(self, config: ForecastConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.config.ensure_directories()

    def run(
        self,
        input_path: str | Path,
        hospital_id: str,
        model_name: str | None = None,
    ) -> TrainingResult:
        model_name = model_name or self.config.default_model_name

        required_columns = [
            self.config.timestamp_col,
            self.config.group_col,
            self.config.target_col,
            *self.config.feature_columns,
        ]

        df = load_dataset(
            path=input_path,
            timestamp_col=self.config.timestamp_col,
            required_columns=required_columns,
        )

        validate_required_columns(df, required_columns)
        validate_no_empty_group(df, self.config.group_col)
        validate_target_not_all_missing(df, self.config.target_col)

        hospital_df = self._select_hospital_timeseries(df, hospital_id)
        hospital_df = hospital_df.sort_values(self.config.timestamp_col).reset_index(drop=True)

        validate_weekly_frequency(hospital_df, self.config.timestamp_col)

        keep_columns = [
            self.config.timestamp_col,
            self.config.group_col,
            self.config.target_col,
            *self.config.feature_columns,
        ]
        hospital_df = hospital_df[keep_columns].copy()

        if self.config.use_lag_features:
            hospital_df = add_lag_features(
                hospital_df,
                target_col=self.config.target_col,
                lag_periods=self.config.lag_periods,
                rolling_window=self.config.rolling_window,
            )
            hospital_df = hospital_df.dropna().reset_index(drop=True)

        train_df, test_df = time_split(hospital_df, self.config.test_size)

        mlflow.set_experiment("icu_forecasting")

        
        with mlflow.start_run(run_name=f"{model_name}_{hospital_id}"):
            mlflow.log_param("hospital_id", str(hospital_id))
            mlflow.log_param("requested_model_name", model_name)
            mlflow.log_param("training_rows", len(train_df))
            mlflow.log_param("test_rows", len(test_df))
            mlflow.log_param("target_col", self.config.target_col)
            mlflow.log_param("frequency", self.config.frequency)
            mlflow.log_param("use_lag_features", self.config.use_lag_features)
            mlflow.log_param("lag_features", self.config.use_lag_features)
            mlflow.log_param("rolling_window", self.config.rolling_window)
            mlflow.log_param("lag_periods", ",".join(map(str, self.config.lag_periods)))

            if model_name == "auto":
                selector = BestModelSelector(
                    metric=self.config.selection_metric,
                    horizon=self.config.test_size,
                    timestamp_col=self.config.timestamp_col,
                    target_col=self.config.target_col,
                    frequency=self.config.frequency,
                )
                selection_result = selector.select_best_model(train_df.copy())
                print("AUTO selected:", selection_result.best_model_name)
                print("AUTO all results:", selection_result.all_results)

                resolved_model_name = selection_result.best_model_name
                model = selection_result.best_model

                mlflow.log_param("model_name", resolved_model_name)
                mlflow.log_param("selection_metric", selection_result.metric_used)
                mlflow.log_metric("best_model_score", selection_result.best_score)

            else:
                resolved_model_name = model_name
                mlflow.log_param("model_name", resolved_model_name)

                model = self._build_model(resolved_model_name)
                self._fit_model(model, train_df)

            predictions = self._predict_model(model, len(test_df))

            metrics = compute_forecast_metrics(
                y_true=test_df[self.config.target_col].tolist(),
                y_pred=predictions,
            )
            results_df = pd.DataFrame({
                "actual": test_df[self.config.target_col].values,
                "predicted": predictions,
            })

            artifact_dir = Path("ml/artifacts")
            artifact_dir.mkdir(parents=True, exist_ok=True)

            csv_path = artifact_dir / f"{hospital_id}_{resolved_model_name}_predictions.csv"
            results_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path))

            plt.figure(figsize=(10, 5))
            plt.plot(results_df["actual"].values, marker="o", label="Actual")
            plt.plot(results_df["predicted"].values, marker="o", label="Predicted")
            plt.title(f"{resolved_model_name} forecast vs actual - hospital {hospital_id}")
            plt.xlabel("Test step")
            plt.ylabel(self.config.target_col)
            plt.legend()
            plt.tight_layout()

            plot_path = artifact_dir / f"{hospital_id}_{resolved_model_name}_forecast_plot.png"
            plt.savefig(plot_path)
            plt.close()

            mlflow.log_artifact(str(plot_path))

            metadata = {
                "metrics": metrics,
                "actuals": test_df[self.config.target_col].tolist(),
                "predictions": predictions,
            }

            if model_name == "auto":
                metadata["selected_model"] = selection_result.best_model_name
                metadata["selection_metric"] = selection_result.metric_used
                metadata["best_score"] = selection_result.best_score
                metadata["all_model_results"] = selection_result.all_results

            return TrainingResult(
                hospital_id=str(hospital_id),
                model_name=resolved_model_name,
                fitted_model=model,
                training_rows=len(train_df),
                artifact_path=None,
                metadata=metadata,
            )

    def _select_hospital_timeseries(self, df: pd.DataFrame, hospital_id: str) -> pd.DataFrame:
        hospital_df = df[df[self.config.group_col].astype(str) == str(hospital_id)].copy()
        if hospital_df.empty:
            raise ValueError(f"No rows found for hospital_id={hospital_id}")
        return hospital_df

    def _build_model(self, model_name: str):
        if model_name == "baseline":
            return BaselineForecaster(target_col=self.config.target_col)
        if model_name == "prophet":
            return ProphetForecaster(
                target_col=self.config.target_col,
                frequency=self.config.frequency,
            )
        if model_name == "sarima":
            return SarimaForecaster(target_col=self.config.target_col)

        raise ValueError(f"Unsupported model_name='{model_name}'")

    def _fit_model(self, model, train_df: pd.DataFrame) -> None:
        if isinstance(model, ProphetForecaster):
            model.fit(train_df, timestamp_col=self.config.timestamp_col)
        else:
            model.fit(train_df)

    def _predict_model(self, model, horizon: int) -> list[float]:
        return model.predict(horizon=horizon)