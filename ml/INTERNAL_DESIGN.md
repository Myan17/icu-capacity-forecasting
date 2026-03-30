# Internal Design Document - ML Module

## 1. Goal

The ML module forecasts short-term ICU occupancy for each hospital and assigns operational risk levels to each future time step.

### Inputs
- historical ICU occupancy values
- timestamps
- hospital identifier
- optional ICU capacity

### Outputs
- future occupancy predictions for the next 24 hours by default
- model metadata
- risk labels: `GREEN`, `YELLOW`, `RED`

---

## 2. Design philosophy

The first version is intentionally **simple, modular, and interpretable**.

We are optimizing for:
- correctness
- readability
- maintainability
- ease of extension

We are **not** optimizing for maximum model complexity in v1.

---

## 3. High-level data flow

1. Load raw hospital time-series data
2. Validate required columns
3. Clean duplicates / invalid values
4. Resample to a fixed interval
5. Train one of the supported models
6. Forecast future ICU occupancy
7. Convert predicted occupancy to risk labels
8. Save forecast results for backend consumption

---

## 4. Module responsibilities

### `config.py`
Defines global settings such as column names, forecast horizon, thresholds, and default model parameters.

### `preprocessing/`
Contains the data preparation layer.

- `loader.py` -> reads CSV files and standardizes types
- `cleaner.py` -> validates rows and removes low-quality input
- `resampler.py` -> normalizes the series to a fixed hourly cadence

### `features/`
Contains optional feature engineering helpers such as hour-of-day and day-of-week.

### `models/`
Each forecasting strategy gets its own wrapper.

- `baseline.py` -> simple benchmark forecast
- `sarima_model.py` -> seasonal ARIMA wrapper
- `prophet_model.py` -> Prophet wrapper

### `pipelines/`
Orchestrates full workflows.

- `training_pipeline.py` -> preprocess + fit + save model metadata
- `inference_pipeline.py` -> preprocess + forecast + risk labeling + serialization

### `utils/`
Shared helpers used across the module.

- `metrics.py` -> MAE / RMSE / MAPE
- `risk.py` -> occupancy-to-risk mapping
- `io.py` -> JSON serialization helpers

### Entrypoints
- `train.py` -> CLI for fitting a chosen model
- `predict.py` -> CLI for inference and writing forecast JSON
- `evaluate.py` -> CLI for offline comparison of models

---

## 5. Why the model layer is separated from backend and dashboard

This keeps concerns clean:

- ML code focuses on forecasting logic
- backend focuses on serving data
- dashboard focuses on visualization

That separation makes it easier to:
- test each layer independently
- switch models without changing the API layer
- cache or precompute predictions

---

## 6. Why we start with baseline, SARIMA, and Prophet

### Baseline
Needed to establish a minimum acceptable benchmark.

### SARIMA
Good for interpretable short-term forecasting with repeating hourly patterns.

### Prophet
Good for trend and seasonality with relatively simple training flow.

This progression gives the project a clean, explainable v1 while leaving room for future upgrades like multivariate models or hybrid deep learning.

---

## 7. Forecast-to-risk conversion

The forecast itself predicts occupancy.
The alerting layer translates that into operational meaning.

Example:
- `GREEN` -> below 70% capacity
- `YELLOW` -> 70% to 85%
- `RED` -> above 85%

This keeps forecasting and business logic loosely coupled.

---

## 8. Training vs inference

### Training
Runs offline and may be slower.
It fits a model using historical data and can save metadata about the training run.

### Inference
Runs when we need a forecast for display or alerting.
It should be lightweight and should not retrain the model on every request.

For the current scaffold, some models are recomputed directly for simplicity, but the folder structure is already prepared for persisted artifacts.

---

## 9. Future upgrades

Planned directions for later versions:
- per-hospital hyperparameter tuning
- multivariate forecasting with admissions / discharges / staffing
- uncertainty bands in API responses
- hybrid ARIMA + neural model approaches
- drift monitoring and automated retraining

---

## 10. Summary

This ML codebase is designed to be:
- readable
- modular
- extensible
- practical for a course or prototype system

The initial focus is getting a clean, trustworthy forecasting foundation in place before optimizing for sophistication.
