# ML Module - ICU Capacity Forecasting

This folder contains the machine learning layer for the ICU Capacity Forecasting project.

## Current forecasting strategy

The first version of the ML stack uses:

1. **Baseline forecast** for a sanity-check benchmark
2. **SARIMA** for interpretable classical time-series modeling
3. **Prophet** for trend + seasonality modeling
4. **Risk labeling** on top of predictions for Green / Yellow / Red alerts

## Why this approach

We are intentionally starting with simple, interpretable models before moving to more complex methods like LSTMs or hybrid models. That gives us:

- easier debugging
- cleaner demos
- faster iteration
- stronger explainability for project reviews

## Folder overview

- `preprocessing/` - load, validate, clean, and resample data
- `features/` - optional time-based feature generation
- `models/` - forecast model wrappers
- `pipelines/` - training and inference orchestration
- `utils/` - metrics, risk labeling, and serialization helpers
- `artifacts/` - saved models, metrics, logs, and forecast outputs
- `data/` - raw and processed files used by the ML module

## Expected input schema

### Required columns
- `hospital_id`
- `timestamp`
- `icu_occupied`

### Optional but recommended
- `icu_capacity`

## Supported input formats

- CSV
- Parquet

Parquet support is included so the ML layer can scale better for larger offline testing datasets and work cleanly with DuckDB-based analysis.

## Current data flow

### Training
1. load raw dataset
2. validate schema
3. clean invalid rows
4. resample one hospital series to fixed cadence
5. fit selected model
6. save trained artifact to `ml/artifacts/models/`

### Inference
1. load latest dataset
2. clean and resample hospital history
3. load saved model artifact if it exists
4. otherwise fit a fresh model
5. forecast future occupancy
6. assign risk levels
7. write forecast JSON to `ml/artifacts/forecasts/`

### Evaluation
1. create a holdout split from the hospital time series
2. train requested model(s)
3. forecast the holdout horizon
4. compute error metrics
5. optionally write comparison results to `ml/artifacts/metrics/`

## Example commands

### Train
```bash
python -m ml.train --input ml/data/raw/sample_hospital_a.csv --hospital-id HOSPITAL_A --model baseline
python -m ml.train --input ml/data/raw/sample_hospital_a.csv --hospital-id HOSPITAL_A --model sarima
python -m ml.train --input ml/data/raw/sample_hospital_a.csv --hospital-id HOSPITAL_A --model prophet