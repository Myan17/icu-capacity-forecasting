# ICU Capacity Forecasting Platform

A full-stack system for predicting short-term ICU capacity risk and surfacing it in an operational dashboard. Combines a FastAPI backend, production ML forecasting pipeline, load testing infrastructure, and a multi-page React UI.

**Deployed:** EC2 (application) + S3 (artifact storage) + SQLite (operational DB + MLflow tracking)

---

## Architecture

```
frontend/          React + Vite — Dashboard, Analytics, Alerts, Settings
backend/           FastAPI + SQLAlchemy — REST API, risk logic, forecast service
lambdas/forecast/  AWS Lambda — serverless forecast handler (moto-tested)
lambdas/ingest/    AWS Lambda — Great-Expectations validated ingestion
lambdas/api/       AWS Lambda — Mangum + FastAPI read API for the dashboard
ml/                Baseline, SARIMA, Prophet — training, evaluation, inference
ml/model_selection ParallelBestModelSelector  — data-parallel candidate fitting (L3)
ml/pipelines/      StagedPipeline / AutoPipelineOptimizer / multiproc inference (L3)
ml/per_hospital/   PerHospitalTuner — non-IID hyperparameter search (L4)
ml/federated/      FederatedClient / Aggregator / FedATCoordinator (L4)
shared/            hospital_tiers / priority_queue / workload_metrics (L1 + L2)
tests/             pytest suite — unit, moto integration, benchmark, CI coverage
tests/load/        Locust load test + MLflow reporter
scripts/           benchmark_models.py, run_training.py, seed_s3.py,
                   simulate_cluster.py (L2), run_federated.py (L4),
                   benchmark_pipelines.py (L3 auto-opt)
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

Locust simulates concurrent hospital users hitting all four core endpoints with realistic task weights (snapshots 5×, forecasts 4×, alerts 3×, run-forecast 1×). On completion, an MLflow event hook aggregates per-hospital metrics and logs them as a single experiment run.

**Stepped load test** (runs at escalating concurrency, auto-increments CSV run number):
```bash
./tests/load/run_stepped.sh                        # default: 10 → 20 → 30 → 50 users, 120s each
USER_STEPS="20 50" RUN_TIME=60s ./tests/load/run_stepped.sh   # custom
```

**Single run:**
```bash
locust -f tests/load/locustfile.py \
  --host https://52-15-187-251.nip.io:8000 \
  --users 10 --spawn-rate 2 \
  --run-time 120s --headless \
  --csv reports/run_N
```

**Compare all runs:**
```bash
python tests/load/compare_runs.py           # all runs
python tests/load/compare_runs.py --last 4  # last 4 only
```

MLflow logs per-hospital: `p50_ms`, `p95_ms`, `p99_ms`, `failure_rate`, `rps`  
MLflow logs summary: `overall_p95_ms`, `overall_failure_rate`, `overall_rps`

#### Capacity findings

Stepped tests (runs 7–10) against the EC2 single-worker backend revealed a clear breakpoint at ~30 concurrent users:

| Users | Requests | Failures | Median | P95    | RPS  |
|-------|----------|----------|--------|--------|------|
| 10    | 584      | 0 (0%)   | 27 ms  | 120 ms | 4.9  |
| 20    | 1162     | 0 (0%)   | 27 ms  | 150 ms | 9.7  |
| 30    | 726      | 0 (0%)   | 880 ms | 6.7 s  | 6.1  |
| 50    | 571      | 22 (4%)  | 6.7 s  | 16 s   | 4.9  |

**Root cause:** `POST /forecast/{id}` runs ML model selection, fitting, and Monte Carlo CI — 100–300 ms of CPU-bound work. On a single-worker async server this blocked the event loop, queuing all concurrent requests behind each inference call. At 30+ users the queue never drained, causing congestion collapse (RPS *fell* as concurrency rose).

**Fix — `run_in_executor` (`backend/app/api/routes.py`):** The `run_forecast` handler is now `async` and offloads the `generate_forecast()` call to asyncio's default thread pool:

```python
loop = asyncio.get_running_loop()
forecast_steps = await loop.run_in_executor(
    None, service.generate_forecast, df, latest_capacity
)
```

The event loop stays free to serve snapshot and alert reads while inference runs in a background thread. numpy/scipy release the GIL during computation so threads execute in true parallel on multi-core instances. DB reads (before) and DB writes (after) stay on the main event loop — they're fast SQLite operations unaffected by the move.

**Next step — full async architecture:** The correct production fix is to decouple `POST /forecast` from the request cycle entirely: accept the call, enqueue to SQS, return `202 Accepted`, and let the forecast Lambda compute and persist the result asynchronously. `GET /forecasts/{id}` then always serves pre-computed results with ~27 ms latency. This is the Lambda refactor targeted in Phase 3.

### Test Suite

```
tests/test_forecast_utils.py                11 tests — breach_prob, risk_label, mc_ci (pure math)
tests/test_forecast_handler.py               5 tests — Lambda handler via moto (DynamoDB + S3 mocked)
tests/test_forecast_handler_extensions.py    4 tests — batch / priority / fallback / metadata
tests/test_benchmark_smoke.py                3 tests — benchmark script returns valid RMSE/MAE
tests/test_forecast_ci_coverage.py           2 tests — MC CI achieves ≥90% coverage
tests/test_ingest_handler.py                 6 tests — ingestion Lambda
tests/test_ingest_validation.py             13 tests — data validation suite
tests/test_hospital_tiers.py                14 tests — workload + TiFL tier classifiers
tests/test_priority_queue.py                 5 tests — Borg-style scheduler + LPT packing
tests/test_workload_metrics.py               4 tests — CPU/IO/comm profiler
tests/test_parallel_selector.py              6 tests — parallel selector + staged pipeline + auto-opt
tests/test_per_hospital_tuner.py             3 tests — per-hospital hyperparameter tuner
tests/test_federated.py                      8 tests — FedAvg / weighted / FedAvgM / FedAT / TiFL
```

All 83 tests pass (3 environment-conditional benchmarks skipped when Prophet/SARIMA fits exceed CI budget).

---

## Distributed-systems extensions (Lectures 1–4)

The platform implements four research patterns from the cloud / distributed-ML / federated-learning literature on top of the same forecast service. Every feature has unit tests and is exercised by either the production handler or a CLI demo.

### Lecture 1 — Cloud Resource Management (MOS-style)

| Feature | Where |
|---|---|
| Workload-aware micro-services (ingest = I/O, forecast = CPU, API = latency) | `template.yaml` per-function `Tags`/`Environment` |
| Dynamic Lambda sizing per hospital tier (memory + timeout + MC samples) | `Mappings.WorkloadTierMap` + `!FindInMap` in `template.yaml`; `shared/hospital_tiers.py` |
| Hospital workload classification (small / medium / large) | `shared/hospital_tiers.py::classify_workload` |
| Per-pipeline CPU vs I/O vs comm telemetry | `shared/workload_metrics.py::WorkloadProfiler` (emits `WallMs_Cpu`, `WallMs_Io`, `WallMs_Comm` to CloudWatch) |

### Lecture 2 — Borg / Kubernetes

| Feature | Where |
|---|---|
| Priority-based scheduling (high-risk hospitals first) | `shared/priority_queue.py::PriorityScheduler`, used by `lambdas/forecast/handler.py::_prioritise_hospitals` |
| Batch forecasting (multi-hospital per Lambda invoke) | `lambdas/forecast/handler.py::lambda_handler` accepts `{"hospital_ids": [...]}` |
| Retry + fallback to `BaselineForecaster` when SARIMA/Prophet fail | `lambdas/forecast/handler.py::_select_model_with_fallback` (emits `ModelFallbackUsed` metric) |
| Request grouping + execution pooling (LPT bin-packing) | `shared/priority_queue.py::ExecutionPool` |
| Multi-node cluster simulator | `python -m scripts.simulate_cluster --hospitals 20 --nodes 4 --rounds 3` |

### Lecture 3 — Distributed ML (Alpa-style)

| Feature | Where |
|---|---|
| Data parallelism across candidate models | `ml/model_selection/parallel_selector.py::ParallelBestModelSelector` (thread / process executors) |
| Pipeline parallelism (load → clean → feature → train → predict) | `ml/pipelines/staged_pipeline.py::StagedPipeline` |
| Inter-op + intra-op parallelism in one run | `StagedPipeline(enable_intra_op=True)` |
| Communication-cost measurement (parquet round-trip bytes between stages) | `StageResult.bytes_in / bytes_out` |
| Latency + bottleneck classification | `shared/workload_metrics.py::detect_bottleneck` |
| GPU-ready / multi-process inference path | `ml/pipelines/multiproc_inference.py::predict_batch(backend="auto")` |
| Auto-optimizer that benchmarks strategies and picks the fastest acceptable one | `ml/pipelines/auto_optimizer.py::AutoPipelineOptimizer`, CLI `python -m scripts.benchmark_pipelines` |

### Lecture 4 — Federated Learning

| Feature | Where |
|---|---|
| Per-hospital local trainer (no raw data leaves the client) | `ml/federated/client.py::FederatedClient` |
| FedAvg / weighted FedAvg / FedAvgM aggregation strategies | `ml/federated/aggregator.py` |
| TiFL tiering by data volume + variance | `shared/hospital_tiers.py::classify_tifl` (Tier 1 / 2 / 3) |
| FedAT — async update cadence per tier (period 1 / 2 / 4) with staleness-decayed weights | `ml/federated/fedat.py::FedATCoordinator` |
| Per-hospital hyperparameter tuning (non-IID baseline) | `ml/per_hospital/tuner.py::PerHospitalTuner` |
| Async forecast cadence on AWS — mid-week refresh for Tier-2 hospitals | `template.yaml::ForecastFunction.Events.Tier2RefreshForecast` |
| Weighted aggregation by `1 / occupancy_variance × num_samples` | `ClientUpdate.fl_weight` and `weighted_fedavg` |

Run the federated simulation locally:

```bash
python -m scripts.run_federated --hospitals 8 --rounds 6 --strategy weighted --mlflow
```

The script logs per-round average loss, tier participation counts, FedAT staleness penalties, and per-hospital RMSE from the classical tuner — all of which can be picked up by MLflow when `--mlflow` is set.

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

**Live dashboard:** https://myan17.github.io/icu-capacity-forecasting/
**Live API:** https://52-15-187-251.nip.io:8000

- **GitHub Pages** serves the React dashboard as a static build, published by
  `.github/workflows/deploy-pages.yml` on every push to `myan`
- **EC2** hosts the FastAPI backend, run by the `hospital-forecasting` systemd
  unit as `uvicorn app.main:app --host 127.0.0.1 --port 8001 --workers 2`
- **Caddy** terminates TLS on port 8000 and reverse-proxies to `127.0.0.1:8001`.
  Pages is served over HTTPS, so a plain-HTTP API would be blocked as mixed
  content; the certificate is issued by Let's Encrypt for the `nip.io` hostname
  that resolves to the instance's public IP, which avoids having to register a
  domain
- **S3** stores trained model artifacts (`.pkl` files, keyed by hospital + model type)
- **MLflow** uses a local SQLite backend (`mlflow.db`) for experiment tracking
- **SQLite** (`app.db`) stores operational data: snapshots, forecasts, alerts

To deploy backend changes to EC2:
```bash
# EC2 terminal
cd ~/icu-capacity-forecasting
git pull
sudo systemctl restart hospital-forecasting
```

To check the backend and its TLS front:
```bash
sudo systemctl status hospital-forecasting caddy
curl https://52-15-187-251.nip.io:8000/health
```
