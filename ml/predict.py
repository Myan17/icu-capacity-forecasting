"""CLI entrypoint for generating forecast JSON outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.config import DEFAULT_CONFIG
from ml.pipelines.inference_pipeline import InferencePipeline
from ml.utils.io import write_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for forecast generation."""
    parser = argparse.ArgumentParser(description="Generate ICU occupancy forecasts")
    parser.add_argument("--input", required=True, help="Path to the hospital CSV or Parquet file")
    parser.add_argument("--hospital-id", required=True, help="Hospital identifier to forecast")
    parser.add_argument(
        "--model",
        default=DEFAULT_CONFIG.default_model_name,
        choices=DEFAULT_CONFIG.enabled_models,
        help="Forecasting model to use",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_CONFIG.forecast_horizon_hours,
        help="Number of future hourly steps to predict",
    )
    parser.add_argument(
        "--capacity",
        type=float,
        default=None,
        help="Optional ICU capacity used for risk labeling",
    )
    parser.add_argument(
        "--artifact-path",
        default=None,
        help="Optional explicit trained model artifact path",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON file path",
    )
    return parser.parse_args()


def main() -> None:
    """Run the inference workflow and save the resulting forecast JSON."""
    args = parse_args()
    DEFAULT_CONFIG.ensure_directories()

    pipeline = InferencePipeline(DEFAULT_CONFIG)
    result = pipeline.run(
        input_path=args.input,
        hospital_id=args.hospital_id,
        model_name=args.model,
        horizon=args.horizon,
        capacity=args.capacity,
        artifact_path=Path(args.artifact_path) if args.artifact_path else None,
    )

    if args.output:
        output_path = Path(args.output)
    else:
        filename = DEFAULT_CONFIG.forecast_filename_template.format(
            hospital_id=args.hospital_id,
            model_name=args.model,
        )
        output_path = DEFAULT_CONFIG.forecast_dir / filename

    write_json(result.to_payload(), output_path)
    print(f"Forecast written to: {output_path}")


if __name__ == "__main__":
    main()