from __future__ import annotations

from pathlib import Path
import pandas as pd

from ml.preprocessing.loader import load_dataset


def test_load_dataset_success(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"

    df = pd.DataFrame(
        {
            "hospital_id": ["100", "100"],
            "timestamp": ["2024-01-07", "2024-01-14"],
            "icu_occupied": [10, 12],
            "icu_capacity": [20, 20],
            "total_beds_7_day_avg": [100, 100],
            "inpatient_beds_used_7_day_avg": [70, 72],
            "total_adult_patients_hospitalized_confirmed_and_suspected_covid_7_day_avg": [5, 4],
            "total_adult_patients_hospitalized_confirmed_covid_7_day_avg": [3, 2],
            "staffed_icu_adult_patients_confirmed_covid_7_day_avg": [1, 1],
        }
    )
    df.to_csv(path, index=False)

    loaded = load_dataset(
        path=path,
        timestamp_col="timestamp",
        required_columns=[
            "hospital_id",
            "timestamp",
            "icu_occupied",
            "icu_capacity",
            "total_beds_7_day_avg",
            "inpatient_beds_used_7_day_avg",
            "total_adult_patients_hospitalized_confirmed_and_suspected_covid_7_day_avg",
            "total_adult_patients_hospitalized_confirmed_covid_7_day_avg",
            "staffed_icu_adult_patients_confirmed_covid_7_day_avg",
        ],
    )

    assert len(loaded) == 2
    assert "timestamp" in loaded.columns