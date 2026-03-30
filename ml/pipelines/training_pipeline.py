"""Training pipeline for ICU forecasting models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ml.config import DEFAULT_CONFIG, ForecastConfig
from ml.models.baseline import BaselineForecaster
from ml.models.prophet_model import ProphetForecaster
from ml.models.sarima_model import SarimaForecaster
from ml.preprocessing.cleaner import clean_timeseries_with_summary
from ml.preprocessing.loader import load_dataset
from ml.preprocessing.resampler import resample_hospital_timeseries
from ml.utils.io import write_json


@dataclass
class TrainingResult:
    """Lightweight container describing what the training stage produced."""

    hospital_id: str
    model_name: str
    fitted_model: Any
    training_rows: int
    artifact_path: str | None
    metadata: Dict[str, Any]


class TrainingPipeline:
    """End-to-end training orchestration for one hospital."""

    def __init__(self, config: ForecastConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.config.ensure_directories()

    def run(
        self,
        input_path: str | Path,
        hospital_id: str,
        model_name: str,
        output_dir: Path | None = None,
    ) -> TrainingResult:
        """Train one model for one hospital and return the fitted object."""
        raw_df = load_dataset(input_path, self.config)
        cleaned_df, cleaning_summary = clean_timeseries_with_summary(raw_df, self.config)
        resampled_df = resample_hospital_timeseries(cleaned_df, hospital_id, self.config)

        series = self._to_training_series(resampled_df)
        model = self._build_model(model_name)
        model.fit(series)

        output_dir = output_dir or self.config.model_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        artifact_filename = self.config.model_filename_template.format(
            hospital_id=hospital_id,
            model_name=model_name,
        )
        artifact_path = output_dir / artifact_filename

        if hasattr(model, "save"):
            model.save(artifact_path)

        metadata = {
            "input_path": str(input_path),
            "resample_frequency": self.config.resample_frequency,
            "history_start": str(series.index.min()),
            "history_end": str(series.index.max()),
            "cleaning_summary": cleaning_summary.to_dict(),
            "artifact_path": str(artifact_path),
        }

        # Optional sidecar metadata file for easier debugging and later API use.
        metadata_path = artifact_path.with_suffix(".metadata.json")
        write_json(
            payload={
                "hospital_id": hospital_id,
                "model_name": model_name,
                "training_rows": int(len(series)),
                "metadata": metadata,
            },
            output_path=metadata_path,
        )

        return TrainingResult(
            hospital_id=hospital_id,
            model_name=model_name,
            fitted_model=model,
            training_rows=len(series),
            artifact_path=str(artifact_path),
            metadata=metadata,
        )

    def _build_model(self, model_name: str):
        """Construct the requested forecasting model."""
        normalized = model_name.strip().lower()
        defaults = self.config.get_model_defaults(normalized)

        if normalized == "baseline":
            return BaselineForecaster(
                rolling_window=int(defaults["rolling_window"]),
                strategy="rolling_mean",
            )

        if normalized == "sarima":
            return SarimaForecaster(
                order=defaults["order"],
                seasonal_order=defaults["seasonal_order"],
            )

        if normalized == "prophet":
            return ProphetForecaster(
                daily_seasonality=bool(defaults["daily_seasonality"]),
                weekly_seasonality=bool(defaults["weekly_seasonality"]),
                yearly_seasonality=bool(defaults["yearly_seasonality"]),
                frequency=str(defaults["frequency"]),
            )

        raise ValueError(f"Unsupported model_name='{model_name}'")

    def _to_training_series(self, resampled_df: pd.DataFrame) -> pd.Series:
        """Convert the resampled frame into the series shape models expect."""
        series = resampled_df.set_index(self.config.timestamp_column)[self.config.target_column]
        series.index = pd.to_datetime(series.index)
        series = series.sort_index().astype(float)
        series = series.asfreq(self.config.resample_frequency)
        return series