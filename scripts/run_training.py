from __future__ import annotations

import argparse

from ml.pipelines.training_pipeline import TrainingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run weekly ICU forecasting pipeline")
    parser.add_argument("--input", required=True, help="Path to cleaned CSV dataset")
    parser.add_argument("--hospital-id", required=True, help="Hospital ID to train on")
    parser.add_argument(
        "--model",
        choices=["baseline", "prophet", "sarima", "auto"],
        default="auto",
    )
    args = parser.parse_args()

    pipeline = TrainingPipeline()
    result = pipeline.run(
        input_path=args.input,
        hospital_id=args.hospital_id,
        model_name=args.model,
    )

    print("Training complete")
    print("Hospital ID:", result.hospital_id)
    print("Model:", result.model_name)
    print("Training rows:", result.training_rows)
    print("Metrics:", result.metadata["metrics"])
    print("Actuals:", result.metadata["actuals"])
    print("Predictions:", result.metadata["predictions"])


if __name__ == "__main__":
    main()