# Load Test Visualization — Design Spec

**Date:** 2026-04-24
**Goal:** Load test the deployed hospital ICU forecasting system across 5–10 hospitals using Locust, log results to MLflow, and visualize per-hospital performance in the React AnalyticsPage.

---

## Architecture

```
Locust (tests/load/locustfile.py)
  │  spawns 1 user per hospital (up to 10)
  │  hits: POST /forecast/{id}, GET /forecasts/{id},
  │        GET /snapshots/latest/{id}, GET /alerts/{id}
  │
  ├─► Locust built-in UI (live during run — RPS, latency, failures)
  │
  └─► MLflow reporter (tests/load/mlflow_reporter.py)
        fires on run end via Locust quitting event hook
        logs per-hospital p50/p95/p99 latency + error rate
        │
        ▼
      mlruns/ (local MLflow store — already exists)
        │
        ├─► mlflow ui  (free cross-run comparison UI)
        │
        └─► GET /load-test/latest + GET /load-test/runs
              new FastAPI routes reading from mlruns/
              │
              ▼
            React AnalyticsPage (currently empty placeholder)
```

---

## Component Specs

### 1. `tests/load/locustfile.py`

- One `HospitalUser` Locust class; hospital IDs (`H001`–`H010`) assigned round-robin via user index
- Task weights mirror real traffic:

| Task | Endpoint | Weight |
|------|----------|--------|
| Read snapshot | `GET /snapshots/latest/{id}` | 5 |
| Read forecasts | `GET /forecasts/{id}` | 4 |
| Read alerts | `GET /alerts/{id}` | 3 |
| Trigger forecast | `POST /forecast/{id}` | 1 |

- `wait_time = between(1, 3)` seconds between tasks
- On `init` event: stores `start_time` for duration tracking
- On `quitting` event: calls `mlflow_reporter.log_run(environment)`

**Run command:**
```bash
locust -f tests/load/locustfile.py \
  --host http://<ec2-url> \
  --users 10 --spawn-rate 1 \
  --run-time 120s --headless \
  --csv reports/run_$(date +%Y%m%d_%H%M%S)
```

---

### 2. `tests/load/mlflow_reporter.py`

- Reads Locust's `stats` object after run completes
- Groups request stats by hospital ID (parsed from URL path)
- Logs one MLflow run to the `hospital-load-test` experiment with:

**Per-hospital metrics** (one value per hospital, keyed by `hospital_id`):
- `{hospital_id}_p50_ms`
- `{hospital_id}_p95_ms`
- `{hospital_id}_p99_ms`
- `{hospital_id}_failure_rate`
- `{hospital_id}_rps`

**Run-level tags:**
- `num_hospitals`
- `run_duration_s`
- `host`
- `timestamp`

**Run-level summary metrics:**
- `overall_p95_ms` — mean p95 across all hospitals
- `overall_failure_rate`
- `overall_rps`

Reads/writes directly from `mlruns/` via `mlflow` SDK — no MLflow server required.

---

### 3. `backend/app/api/routes.py` — 2 new routes

**`GET /load-test/latest`**
- Calls `mlflow.search_runs(experiment_names=["hospital-load-test"], max_results=1, order_by=["start_time DESC"])`
- Returns per-hospital breakdown as JSON:
```json
{
  "run_id": "abc123",
  "timestamp": "2026-04-24T22:00:00",
  "num_hospitals": 10,
  "run_duration_s": 120,
  "hospitals": [
    {"hospital_id": "H001", "p50_ms": 42, "p95_ms": 118, "p99_ms": 210, "failure_rate": 0.01, "rps": 2.4},
    ...
  ]
}
```
- Returns `{"error": "no runs found"}` with 404 if MLflow has no runs yet

**`GET /load-test/runs`**
- Returns last 20 MLflow runs with summary metrics only (for trend chart)
```json
[
  {"run_id": "abc123", "timestamp": "...", "overall_p95_ms": 118, "overall_failure_rate": 0.01, "num_hospitals": 10},
  ...
]
```

---

### 4. `frontend/src/pages/AnalyticsPage.jsx`

Replaces current placeholder. Two tabs — no new npm packages needed (Recharts + Axios already installed).

**Tab 1 — Latest Run**
- Data source: `GET /load-test/latest`
- `BarChart` (Recharts): x = hospital_id, y = p95_ms — shows slowest hospital at a glance
- Stat cards below chart: overall RPS, overall failure rate, run timestamp, num hospitals

**Tab 2 — Run History**
- Data source: `GET /load-test/runs`
- `LineChart` (Recharts): x = timestamp, y = overall_p95_ms — one line showing trend across runs
- Lets you see if a redeployment improved or degraded p95 latency

**Data flow:**
```
AnalyticsPage mounts
  → axios.get("/load-test/latest")  → Tab 1 state
  → axios.get("/load-test/runs")    → Tab 2 state
Loading/error states handled with existing LoadingState + ErrorState components
```

---

## File Map

| Action | Path |
|--------|------|
| Create | `tests/load/locustfile.py` |
| Create | `tests/load/mlflow_reporter.py` |
| Modify | `backend/app/api/routes.py` |
| Modify | `frontend/src/pages/AnalyticsPage.jsx` |

---

## Out of Scope

- Real-time streaming of Locust metrics to the frontend during a run (Locust UI handles this live)
- Grafana / InfluxDB integration
- Storing load test results in DynamoDB or SQLite
- Auth on the `/load-test/*` routes (internal tooling)
