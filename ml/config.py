from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ForecastConfig:
    """Configuration for ICU forecasting pipeline."""

    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    model_dir: Path = Path("artifacts/models")
    metrics_dir: Path = Path("artifacts/metrics")
    plots_dir: Path = Path("artifacts/plots")

    # Dataset schema for cleaned_hhs_ml_ready.csv
    timestamp_col: str = "timestamp"
    group_col: str = "hospital_id"
    target_col: str = "icu_occupied"

    feature_columns: list[str] = field(
        default_factory=lambda: [
            "total_beds_7_day_avg",
            "inpatient_beds_used_7_day_avg",
            "total_adult_patients_hospitalized_confirmed_and_suspected_covid_7_day_avg",
            "total_adult_patients_hospitalized_confirmed_covid_7_day_avg",
            "icu_capacity",
            "staffed_icu_adult_patients_confirmed_covid_7_day_avg",
        ]
    )

    optional_columns: list[str] = field(default_factory=lambda: ["state"])

    frequency: str = "W"
    input_is_preaggregated: bool = True

    forecast_horizon: int = 8
    test_size: int = 8
    selection_metric: str = "rmse"

    use_lag_features: bool = True
    lag_periods: list[int] = field(default_factory=lambda: [1, 2, 4])
    rolling_window: int = 4

    default_model_name: str = "auto"

    def ensure_directories(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = ForecastConfig()