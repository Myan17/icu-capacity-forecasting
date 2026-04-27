# ICU Capacity Forecasting Platform

A full-stack system for predicting short-term ICU capacity risk and surfacing it in an operational dashboard. Combines a FastAPI backend, production ML forecasting pipeline, load testing infrastructure, and a multi-page React UI.

**Deployed:** EC2 (application) + S3 (artifact storage) + SQLite (operational DB + MLflow tracking)

---

## Architecture

```
frontend/          React + Vite — Dashboard, Analytics, Alerts, Settings
backend/           FastAPI + SQLAlchemy — REST API, risk logic, forecast service
lambdas/forecast/  AWS Lambda — serverless forecast handler (moto-tested)
ml/                Baseline, SARIMA, Prophet — training, evaluation, inference
tests/             pytest suite — unit, moto integration, benchmark, CI coverage
tests/load/        Locust load test + MLflow reporter
scripts/           benchmark_models.py, run_training.py, seed_s3.py
```

---

## What's Implemented

### ML Forecasting Pipeline

The live API auto-selects the best model per hospital from three candidates:

| Model | Selection criterion |
|---|---|
| **Baseline** | Moving-average extrapolation — always available |
| **SARIMA** | Chosen when seasonal patterns detected in history |
| **Prophet** | Chosen when sufficient history exists and trend is non-linear |

Each forecast includes:
- **Monte Carlo confidence intervals** (500 samples, residual-bootstrapped)
- **Breach probability** — P(occupancy > RED threshold) over the forecast horizon
- **Risk label** — `GREEN` / `YELLOW` / `RED` per forecast step

Model artifacts are stored in S3 and loaded at inference time. If no artifact exists, the system falls back to Baseline.

### Benchmark Results

Run on real hospital snapshot data across all three models:

| Model | MAE | RMSE | MAPE | SMAPE |
|---|---:|---:|---:|---:|
| Baseline | 1.611 | 1.893 | 1.867 | 1.886 |
| Prophet | 2.676 | 3.301 | 3.131 | 3.057 |
| SARIMA | 4.081 | 4.465 | 4.733 | 4.871 |

Baseline outperforms advanced models on the current dataset, which is expected with weekly-cadence coarse-grained ICU data. The auto-selector accounts for this per-hospital.

### Risk Logic

| Level | Condition |
|---|---|
| `GREEN` | Occupancy ratio < 75% |
| `YELLOW` | Occupancy ratio ≥ 75% |
| `RED` | Occupancy ratio ≥ 90% |

### REST API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/hospitals` | List all hospitals with snapshot data |
| POST | `/snapshots` | Ingest an ICU snapshot |
| GET | `/snapshots/latest/{hospital_id}` | Latest snapshot for a hospital |
| POST | `/forecast/{hospital_id}` | Run forecast + store results + generate alerts |
| GET | `/forecasts/{hospital_id}` | Retrieve stored forecast with CI bands |
| GET | `/alerts/{hospital_id}` | Alerts for a specific hospital |
| GET | `/alerts` | All active alerts across all hospitals |
| GET | `/load-test/latest` | Latest Locust run — per-hospital p50/p95/p99/RPS/failure rate |
| GET | `/load-test/runs` | Last 20 Locust runs — summary metrics for trend view |

### Frontend Pages

- **Dashboard** — hospital selector, KPI cards (occupancy, capacity, staffing, admissions), forecast chart with CI bands and threshold lines, active alerts panel, breach probability + model name
- **Analytics** — load test results: stat cards (overall p95, RPS, failure rate, hospitals tested), per-hospital p95 bar chart (latest run), p95 trend line + history table (run history tab)
- **Alerts** — all active alerts across all hospitals, color-coded by severity (RED/YELLOW), sortable table
- **Settings** — current system configuration (thresholds, model selector, horizon, MC samples)

### Load Testing

Locust simulates one user per hospital hitting all four core endpoints with realistic task weights (snapshots 5×, forecasts 4×, alerts 3×, run-forecast 1×). On run completion, an MLflow event hook aggregates per-hospital metrics and logs them as a single experiment run.

```bash
# Mac terminal — run against EC2 backend
locust -f tests/load/locustfile.py \
  --host http://52.15.187.251:8000 \
  --users 10 --spawn-rate 1 \
  --run-time 120s --headless \
  --csv reports/run_$(date +%Y%m%d_%H%M%S)
```

MLflow logs per-hospital: `p50_ms`, `p95_ms`, `p99_ms`, `failure_rate`, `rps`  
MLflow logs summary: `overall_p95_ms`, `overall_failure_rate`, `overall_rps`

### Test Suite

```
tests/test_forecast_utils.py       11 tests — breach_prob, risk_label, mc_ci (pure math)
tests/test_forecast_handler.py      5 tests — Lambda handler via moto (DynamoDB + S3 mocked)
tests/test_benchmark_smoke.py       3 tests — benchmark script returns valid RMSE/MAE
tests/test_forecast_ci_coverage.py  2 tests — MC CI achieves ≥90% coverage (Gaussian + uniform noise)
tests/test_ingest_handler.py        ingestion Lambda tests
tests/test_ingest_validation.py     data validation suite
```

All 21 phase-3 tests pass:
```
21 passed, 1 warning in 3.86s
```

---

## Running Locally

### Backend

```bash
# Mac terminal
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
# Mac terminal
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### Run forecast for a hospital

```bash
curl -s -X POST http://localhost:8000/forecast/010001
curl -s http://localhost:8000/forecasts/010001
curl -s http://localhost:8000/alerts/010001
```

### Train models

```bash
python scripts/run_training.py
```

### Run benchmark

```bash
python scripts/benchmark_models.py
```

---

## Cloud Deployment

- **EC2** hosts the FastAPI backend (`http://52.15.187.251:8000`)
- **S3** stores trained model artifacts (`.pkl` files, keyed by hospital + model type)
- **MLflow** uses a local SQLite backend (`mlflow.db`) for experiment tracking
- **SQLite** (`app.db`) stores operational data: snapshots, forecasts, alerts

To deploy backend changes to EC2:
```bash
# EC2 terminal
cd ~/hospital-forecasting
git pull
sudo systemctl restart hospital-forecasting
```
