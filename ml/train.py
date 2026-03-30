"""CLI entrypoint for training one forecasting model."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.config import DEFAULT_CONFIG
from ml.pipelines.training_pipeline import TrainingPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model training."""
    parser = argparse.ArgumentParser(description="Train an ICU forecasting model")
    parser.add_argument("--input", required=True, help="Path to the hospital CSV or Parquet file")
    parser.add_argument("--hospital-id", required=True, help="Hospital identifier to train on")
    parser.add_argument(
        "--model",
        default=DEFAULT_CONFIG.default_model_name,
        choices=DEFAULT_CONFIG.enabled_models,
        help="Forecasting model to train",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_CONFIG.model_dir),
        help="Directory where trained model artifacts should be stored",
    )
    return parser.parse_args()


def main() -> None:
    """Run the training workflow and print a compact summary."""
    args = parse_args()
    DEFAULT_CONFIG.ensure_directories()

    pipeline = TrainingPipeline(DEFAULT_CONFIG)
    result = pipeline.run(
        input_path=args.input,
        hospital_id=args.hospital_id,
        model_name=args.model,
        output_dir=Path(args.output_dir),
    )

    print("Training completed")
    print(f"hospital_id: {result.hospital_id}")
    print(f"model_name: {result.model_name}")
    print(f"training_rows: {result.training_rows}")
    print(f"artifact_path: {getattr(result, 'artifact_path', None)}")
    print(f"metadata: {result.metadata}")


if __name__ == "__main__":
    main()