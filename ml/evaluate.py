"""CLI entrypoint for offline model evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml.config import DEFAULT_CONFIG
from ml.pipelines.training_pipeline import TrainingPipeline
from ml.preprocessing.cleaner import clean_timeseries_with_summary
from ml.preprocessing.loader import load_dataset
from ml.preprocessing.resampler import resample_hospital_timeseries
from ml.utils.io import write_dataframe_csv, write_json
from ml.utils.metrics import summarize_metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate ICU forecasting models")
    parser.add_argument("--input", required=True, help="Path to the hospital CSV or Parquet file")
    parser.add_argument("--hospital-id", required=True, help="Hospital identifier to evaluate")
    parser.add_argument(
        "--model",
        default=None,
        choices=DEFAULT_CONFIG.enabled_models,
        help="Evaluate one specific model",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Evaluate one or more models",
    )
    parser.add_argument(
        "--write-results",
        action="store_true",
        help="Write evaluation results to artifact files",
    )
    return parser.parse_args()


def main() -> None:
    """Perform a simple holdout evaluation for each requested model."""
    args = parse_args()
    config = DEFAULT_CONFIG
    config.ensure_directories()

    pipeline = TrainingPipeline(config)

    raw_df = load_dataset(args.input, config)
    cleaned_df, cleaning_summary = clean_timeseries_with_summary(raw_df, config)
    resampled_df = resample_hospital_timeseries(cleaned_df, args.hospital_id, config)
    full_series = pipeline._to_training_series(resampled_df)  # acceptable internal reuse here

    horizon = min(config.test_horizon_hours, max(1, len(full_series) // 5))
    if len(full_series) <= horizon:
        raise ValueError(
            f"Not enough data to evaluate hospital '{args.hospital_id}'. "
            f"Need more than {horizon} rows, found {len(full_series)}."
        )

    train_series = full_series.iloc[:-horizon]
    test_series = full_series.iloc[-horizon:]

    requested_models = _resolve_models(args)
    results: list[dict[str, object]] = []

    for model_name in requested_models:
        try:
            model = pipeline._build_model(model_name)
            model.fit(train_series)
            predicted = model.forecast(len(test_series))
            metrics = summarize_metrics(test_series, predicted)

            result_row = {
                "hospital_id": args.hospital_id,
                "model_name": model_name,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "smape": metrics["smape"],
                "train_rows": int(len(train_series)),
                "test_rows": int(len(test_series)),
            }
            results.append(result_row)
            print(f"model={model_name} metrics={metrics}")

        except Exception as exc:  # keep evaluation robust when one model fails
            result_row = {
                "hospital_id": args.hospital_id,
                "model_name": model_name,
                "error": str(exc),
                "train_rows": int(len(train_series)),
                "test_rows": int(len(test_series)),
            }
            results.append(result_row)
            print(f"model={model_name} failed error={exc}")

    if args.write_results:
        _write_results(
            hospital_id=args.hospital_id,
            results=results,
            input_path=args.input,
            horizon=horizon,
            cleaning_summary=cleaning_summary.to_dict(),
        )


def _resolve_models(args: argparse.Namespace) -> list[str]:
    """Resolve model selection from CLI args."""
    if args.model:
        return [args.model]
    if args.models:
        return args.models
    return DEFAULT_CONFIG.enabled_models


def _write_results(
    hospital_id: str,
    results: list[dict[str, object]],
    input_path: str,
    horizon: int,
    cleaning_summary: dict[str, object],
) -> None:
    """Persist evaluation results to JSON and CSV."""
    timestamp = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ")
    base_name = f"{hospital_id}_evaluation_{timestamp}"

    json_path = DEFAULT_CONFIG.metrics_dir / f"{base_name}.json"
    csv_path = DEFAULT_CONFIG.metrics_dir / f"{base_name}.csv"

    payload = {
        "hospital_id": hospital_id,
        "input_path": str(input_path),
        "evaluation_horizon": horizon,
        "cleaning_summary": cleaning_summary,
        "results": results,
    }
    write_json(payload, json_path)

    results_df = pd.DataFrame(results)
    write_dataframe_csv(results_df, csv_path, index=False)


if __name__ == "__main__":
    main()