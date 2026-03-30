"""Inference pipeline for generating forecasts and risk labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ml.config import DEFAULT_CONFIG, ForecastConfig
from ml.models.baseline import BaselineForecaster
from ml.models.prophet_model import ProphetForecaster
from ml.models.sarima_model import SarimaForecaster
from ml.preprocessing.cleaner import clean_timeseries_with_summary
from ml.preprocessing.loader import load_dataset
from ml.preprocessing.resampler import resample_hospital_timeseries
from ml.utils.risk import add_risk_column


@dataclass
class InferenceResult:
    """Structured result returned by the inference pipeline."""

    hospital_id: str
    model_name: str
    generated_at: str
    forecast_df: pd.DataFrame
    metadata: Dict[str, Any]

    def to_payload(self) -> Dict[str, Any]:
        """Convert the inference result into an API-friendly dictionary."""
        return {
            "hospital_id": self.hospital_id,
            "model_name": self.model_name,
            "generated_at": self.generated_at,
            "forecasts": self.forecast_df.to_dict(orient="records"),
            "metadata": self.metadata,
        }


class InferencePipeline:
    """Run forecasting for a single hospital using saved artifacts when available."""

    def __init__(self, config: ForecastConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.config.ensure_directories()

    def run(
        self,
        input_path: str | Path,
        hospital_id: str,
        model_name: str,
        horizon: Optional[int] = None,
        capacity: Optional[float] = None,
        artifact_path: Path | None = None,
    ) -> InferenceResult:
        """Generate a forecast and attach risk labels."""
        horizon_steps = horizon or self.config.forecast_horizon_hours

        raw_df = load_dataset(input_path, self.config)
        cleaned_df, cleaning_summary = clean_timeseries_with_summary(raw_df, self.config)
        resampled_df = resample_hospital_timeseries(cleaned_df, hospital_id, self.config)
        series = self._to_training_series(resampled_df)

        model = self._load_or_fit_model(
            series=series,
            hospital_id=hospital_id,
            model_name=model_name,
            artifact_path=artifact_path,
        )

        raw_forecast = model.forecast(horizon_steps)
        forecast_frame = self._build_forecast_frame(
            hospital_id=hospital_id,
            raw_forecast=raw_forecast,
            model_name=model_name,
            anchor_timestamp=series.index.max(),
        )
        forecast_frame = add_risk_column(forecast_frame, capacity, config=self.config)

        return InferenceResult(
            hospital_id=hospital_id,
            model_name=model_name,
            generated_at=pd.Timestamp.utcnow().isoformat(),
            forecast_df=forecast_frame,
            metadata={
                "input_path": str(input_path),
                "horizon": horizon_steps,
                "capacity": capacity,
                "history_start": str(series.index.min()),
                "history_end": str(series.index.max()),
                "cleaning_summary": cleaning_summary.to_dict(),
                "artifact_path": str(artifact_path) if artifact_path else None,
            },
        )

    def _load_or_fit_model(
        self,
        series: pd.Series,
        hospital_id: str,
        model_name: str,
        artifact_path: Path | None,
    ):
        """Load a saved model if available; otherwise fit a fresh one."""
        resolved_artifact_path = artifact_path
        if resolved_artifact_path is None:
            filename = self.config.model_filename_template.format(
                hospital_id=hospital_id,
                model_name=model_name,
            )
            resolved_artifact_path = self.config.model_dir / filename

        if resolved_artifact_path.exists():
            return self._load_model(model_name, resolved_artifact_path)

        model = self._build_model(model_name)
        model.fit(series)
        return model

    def _build_model(self, model_name: str):
        """Construct a fresh forecasting model."""
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

    def _load_model(self, model_name: str, artifact_path: Path):
        """Load a serialized model artifact by model type."""
        normalized = model_name.strip().lower()

        if normalized == "baseline":
            return BaselineForecaster.load(artifact_path)
        if normalized == "sarima":
            return SarimaForecaster.load(artifact_path)
        if normalized == "prophet":
            return ProphetForecaster.load(artifact_path)

        raise ValueError(f"Unsupported model_name='{model_name}'")

    def _to_training_series(self, resampled_df: pd.DataFrame) -> pd.Series:
        """Convert the resampled frame into the series shape models expect."""
        series = resampled_df.set_index(self.config.timestamp_column)[self.config.target_column]
        series.index = pd.to_datetime(series.index)
        return series.sort_index().astype(float)

    def _build_forecast_frame(
        self,
        hospital_id: str,
        raw_forecast: pd.Series,
        model_name: str,
        anchor_timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        """Normalize model output into a standard forecast table."""
        values = pd.Series(raw_forecast).reset_index(drop=True).astype(float)

        # Anchor forecast timestamps to the latest observed timestamp, not current UTC time.
        freq = self.config.resample_frequency
        start_time = pd.Timestamp(anchor_timestamp) + pd.Timedelta(freq)
        forecast_times = pd.date_range(start=start_time, periods=len(values), freq=freq)

        return pd.DataFrame(
            {
                "hospital_id": hospital_id,
                "model_name": model_name,
                "forecast_time": forecast_times,
                "predicted_icu_occupied": values,
            }
        )