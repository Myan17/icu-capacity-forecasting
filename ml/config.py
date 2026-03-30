"""Central configuration for the ICU forecasting ML module.

This file keeps the project's default settings in one place so training,
inference, and evaluation use the same assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set


@dataclass(frozen=True)
class ForecastConfig:
    """Project-wide settings for preprocessing, training, and inference."""

    # Project paths.
    project_root: Path = Path(__file__).resolve().parent
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    artifact_dir: Path = project_root / "artifacts"
    model_dir: Path = artifact_dir / "models"
    forecast_dir: Path = artifact_dir / "forecasts"
    metrics_dir: Path = artifact_dir / "metrics"
    logs_dir: Path = artifact_dir / "logs"

    # Data schema.
    timestamp_column: str = "timestamp"
    hospital_id_column: str = "hospital_id"
    target_column: str = "icu_occupied"
    capacity_column: str = "icu_capacity"

    # Supported file formats.
    supported_input_formats: Set[str] = field(default_factory=lambda: {".csv", ".parquet"})

    # Resampling / forecasting defaults.
    resample_frequency: str = "1h"
    forecast_horizon_hours: int = 24
    test_horizon_hours: int = 24
    rolling_window_hours: int = 6
    season_length: int = 24  # Hourly data with daily seasonality.

    # Model selection.
    enabled_models: List[str] = field(default_factory=lambda: ["baseline", "sarima", "prophet"])
    default_model_name: str = "baseline"

    # Risk threshold configuration based on predicted occupancy ratio.
    green_threshold: float = 0.70
    yellow_threshold: float = 0.85

    # Business rules.
    enforce_capacity_upper_bound: bool = True
    drop_rows_with_missing_capacity: bool = False

    # SARIMA defaults.
    sarima_order: tuple[int, int, int] = (1, 1, 1)
    sarima_seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 24)

    # Prophet defaults.
    prophet_daily_seasonality: bool = True
    prophet_weekly_seasonality: bool = True
    prophet_yearly_seasonality: bool = False
    prophet_frequency: str = "1h"

    # Output behavior.
    forecast_filename_template: str = "{hospital_id}_{model_name}_forecast.json"
    metrics_filename_template: str = "{hospital_id}_{model_name}_metrics.json"
    model_filename_template: str = "{hospital_id}_{model_name}.pkl"

    def ensure_directories(self) -> None:
        """Create runtime directories if they do not already exist."""
        for path in (
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.artifact_dir,
            self.model_dir,
            self.forecast_dir,
            self.metrics_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def required_columns(self) -> List[str]:
        """Return the minimum required schema for training/inference."""
        return [
            self.hospital_id_column,
            self.timestamp_column,
            self.target_column,
        ]

    def optional_columns(self) -> List[str]:
        """Return optional but useful columns."""
        return [self.capacity_column]

    def risk_thresholds(self) -> Dict[str, float]:
        """Return thresholds in a compact dictionary."""
        return {
            "green": self.green_threshold,
            "yellow": self.yellow_threshold,
        }

    def get_model_defaults(self, model_name: str) -> Dict[str, object]:
        """Return default settings for a specific model."""
        if model_name == "baseline":
            return {"rolling_window": self.rolling_window_hours}
        if model_name == "sarima":
            return {
                "order": self.sarima_order,
                "seasonal_order": self.sarima_seasonal_order,
            }
        if model_name == "prophet":
            return {
                "daily_seasonality": self.prophet_daily_seasonality,
                "weekly_seasonality": self.prophet_weekly_seasonality,
                "yearly_seasonality": self.prophet_yearly_seasonality,
                "frequency": self.prophet_frequency,
            }
        raise ValueError(f"Unsupported model_name='{model_name}'")


DEFAULT_CONFIG = ForecastConfig()
DEFAULT_CONFIG.ensure_directories()