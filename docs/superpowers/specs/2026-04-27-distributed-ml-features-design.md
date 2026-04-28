# Distributed ML Features Design
**Date:** 2026-04-27
**Branch:** myan
**Status:** Approved

---

## Goal

Add four simulated distributed-systems features to the hospital-forecasting ML pipeline, each mapping to a CSCI5980 lecture concept. All implementations are additive — existing code is not replaced, only wrapped or extended.

Target: `"A federated, distributed, workload-aware ICU forecasting system with adaptive tier-based scheduling and parallel ML execution"`

---

## Scope

| Lecture | Feature | Module |
|---|---|---|
| L1 — MOS | Workload Profiler | `ml/workload/profiler.py` |
| L2 — Borg/K8s | Tier Scheduler | `ml/scheduler/priority_queue.py` |
| L3 — Distributed ML | Parallel Executor | `ml/parallel/executor.py` |
| L4 — Federated Learning | Federated Aggregator | `ml/federated/` |
| Stretch | Observability dashboard | Backend route + frontend panel |

---

## Architecture Overview

All four modules sit between the existing SQS trigger and the existing DynamoDB writes. The execution order per forecast invocation:

```
SQS message → ForecastLambda
    │
    ▼
[L1] WorkloadProfiler
     classify hospital → tier (SMALL / MEDIUM / LARGE)
     assign compute budget (priority_score, mc_samples)
    │
    ▼
[L2] TierScheduler
     sort by (tier ASC, staleness DESC)
     bin-pack SMALL hospitals (≤5 per batch)
    │
    ▼
[L3] ParallelExecutor
     ThreadPoolExecutor(max_workers=4)
     3-stage DAG per hospital: preprocess → train → predict
    │
    ▼
[L4] FederatedAggregator  (fires after all-hospitals path only)
     local weights → FedAvg → global_model.joblib → S3
    │
    ▼
Existing DynamoDB writes (unchanged)
```

**Key constraint:** everything runs inside the existing `ForecastFunction` Lambda. No new Lambda functions are introduced.

---

## Module 1 — WorkloadProfiler (L1: MOS)

**File:** `ml/workload/profiler.py`

**Purpose:** Classify each hospital by compute workload so downstream components can allocate resources proportionally.

**Inputs:** `hospital_id: str`, `df: pd.DataFrame` (snapshot history, up to 52 rows)

**Output:** `WorkloadProfile` dataclass

```
WorkloadProfile:
    hospital_id: str
    tier: Tier           # LARGE | MEDIUM | SMALL
    priority_score: float
    mc_samples: int
    reason: str
```

**Tier logic:**

| Tier | Condition | Priority | MC samples |
|---|---|---|---|
| LARGE | rows ≥ 30 AND variance(icu_occupied) > 50 | 1 (highest) | 1000 |
| MEDIUM | rows 15–29 OR variance 20–50 | 2 | 500 |
| SMALL | rows < 15 OR variance < 20 | 3 | 200 |

**priority_score formula:** `rows * 0.6 + variance_normalized * 0.4`

**Caching:** Profiles cached in-memory per Lambda container warm start (dict keyed by hospital_id). Profiles also written as a DynamoDB attribute on the hospital's snapshot PK for frontend display.

**Integration:** `_run_forecast_for_hospital()` calls `WorkloadProfiler.profile()` first and passes `profile.mc_samples` to `_mc_forecast()` instead of the hardcoded `MC_SAMPLES = 500`.

---

## Module 2 — TierScheduler (L2: Borg/Kubernetes)

**File:** `ml/scheduler/priority_queue.py`

**Purpose:** Replace the flat `for hid in hospital_ids` loop with a priority-aware scheduler that bin-packs small hospitals and handles stragglers.

**Inputs:** `list[WorkloadProfile]`

**Output:** `list[JobBatch]`

```
JobBatch:
    hospital_ids: list[str]   # 1 for LARGE/MEDIUM, up to 5 for SMALL
    tier: Tier
    priority: int
    estimated_duration_s: float
```

**Scheduling rules:**
1. Sort all hospitals by `(tier ASC, last_forecast_age DESC)` — LARGE+stale executes first
2. Bin-pack SMALL hospitals: group up to 5 per `JobBatch` (share one thread)
3. MEDIUM and LARGE: one `JobBatch` per hospital

**Straggler simulation:** if a SMALL batch runtime exceeds 2× `estimated_duration_s`, remaining hospitals in that batch return partial results and are flagged as `status: deferred` in the response.

**Metrics emitted:** `SchedulerQueueDepth` (CloudWatch), per-tier job counts.

---

## Module 3 — ParallelExecutor (L3: Distributed ML)

**File:** `ml/parallel/executor.py`

**Purpose:** Run scheduled job batches concurrently using a thread pool, with each hospital's pipeline represented as a 3-stage DAG.

**Inputs:** `list[JobBatch]`, `max_workers: int = 4`

**Output:** `list[JobResult]`

```
JobResult:
    hospital_ids: list[str]
    status: str
    wall_clock_s: float
    stage_durations: dict[str, float]   # preprocess / train / predict
    parallelism_efficiency: float
```

**Pipeline DAG (per hospital, 3 stages in order):**

| Stage | Function | Dependency |
|---|---|---|
| preprocess | `_items_to_dataframe()` | none |
| train | `_select_model()` | preprocess |
| predict | `_generate_forecast()` | train |

The DAG is represented as an `OrderedDict[str, Callable]` — simple but sufficient for a simulated pipeline graph.

**Execution model:**
- `ThreadPoolExecutor(max_workers=4)` — one Future per `JobBatch`
- Intra-batch hospitals (SMALL bins) run sequentially within their thread
- Inter-batch jobs run in parallel across threads

**Metrics emitted:**
- `wall_clock_time` — total elapsed for the full batch set
- `per_stage_duration` — time per DAG stage averaged across hospitals
- `parallelism_efficiency = sum(serial_times) / (max_workers * wall_clock)` — value < 1 means good parallelism

---

## Module 4 — FederatedAggregator (L4: Federated Learning / TiFL / FedAT)

**Files:**
```
ml/federated/__init__.py
ml/federated/local_trainer.py    — per-hospital local model + weight extraction
ml/federated/aggregator.py       — FedAvg implementation
ml/federated/round_tracker.py    — round number, convergence metrics
```

**Simulation approach:** Each `hospital_id` is a "client." Raw data never leaves the hospital (already true — only aggregated weights are shared). The global model is stored in S3.

### FL Round Flow

**Local phase (per hospital):**
1. Download `fl/global_model.joblib` from S3 (if round > 0; skip on round 0)
2. `LocalTrainer.train(hospital_df)` — fine-tune on hospital's own data using `TrainingPipeline`
3. Extract "weights" from fitted model:
   - Baseline → `{"last_value": float}`
   - SARIMA → `{"ar_params": list[float], "ma_params": list[float]}`
   - Prophet → `{"k": float, "m": float, "changepoint_prior_scale": float}`
4. Upload `fl/round_{N}/{hospital_id}_weights.json` to S3

**Aggregation phase (fires at end of all-hospitals path):**
5. Download all `fl/round_{N}/*.json` weight files
6. FedAvg: `global_weight[k] = Σ (n_i / N_total) * local_weight_i[k]` where `n_i` = training row count for hospital `i`
7. Reconstruct global model from averaged weights
8. Upload to `fl/global_model.joblib`
9. Write round record to DynamoDB:
   ```
   pk: FL#GLOBAL  sk: ROUND#{N}
   round_id, num_clients, avg_rmse, convergence_delta, timestamp
   ```

### Tier participation (TiFL simulation)

| Tier | Participation frequency |
|---|---|
| LARGE (Tier 1) | Every round |
| MEDIUM (Tier 2) | Every 2 rounds |
| SMALL (Tier 3) | Every 4 rounds |

`RoundTracker.should_participate(hospital_id, tier, round_n)` returns bool. Non-participating hospitals skip the local training phase and reuse the previous round's global model for inference.

### Async handling (FedAT simulation)

LARGE hospitals do not wait for SMALL hospitals to finish before aggregation begins. `FederatedAggregator.aggregate(round_n, min_clients=N_large)` runs as soon as all LARGE-tier clients have uploaded weights, then incorporates available MEDIUM/SMALL weights within a 30-second window.

---

## Data Model Additions

| Table | New attribute | Purpose |
|---|---|---|
| SnapshotsTable (hospital PK) | `workload_tier`, `priority_score` | Display in frontend |
| ForecastsTable | `parallelism_efficiency`, `fl_round` | Analytics |
| New: `FLRoundsTable` | `pk=FL#GLOBAL sk=ROUND#{N}` | FL convergence history |

`FLRoundsTable` is a new DynamoDB table added to `template.yaml`.

---

## Handler Integration (minimal changes to existing code)

In `lambdas/forecast/handler.py`:

1. `_resolve_hospital_ids()` → unchanged
2. Replace `for hid in hospital_ids` loop with:
   ```python
   profiles = [WorkloadProfiler().profile(hid, df) for hid, df in snapshots]
   batches = TierScheduler().schedule(profiles)
   results = ParallelExecutor().run(batches, run_fn=_run_forecast_for_hospital)
   FederatedAggregator().aggregate_if_due(round_tracker)
   ```
3. `_run_forecast_for_hospital(hid)` → accepts optional `mc_samples: int` kwarg (default unchanged)

---

## Testing Strategy

| Module | Test file | Approach |
|---|---|---|
| WorkloadProfiler | `tests/test_workload_profiler.py` | Synthetic DataFrames of known row count + variance; assert tier assignment |
| TierScheduler | `tests/test_tier_scheduler.py` | Assert LARGE always precedes SMALL in batch list; assert SMALL bin-packing ≤ 5 |
| ParallelExecutor | `tests/test_parallel_executor.py` | Mock `_run_forecast_for_hospital`; assert wall_clock < sum(serial_times) with workers > 1 |
| FederatedAggregator | `tests/test_federated_aggregator.py` | Mock S3; assert FedAvg output equals weighted mean of synthetic weight dicts |
| RoundTracker | inline in `test_federated_aggregator.py` | Assert SMALL skips rounds 1, 2, 3; participates on round 4 |

All tests are unit tests (no AWS). Moto used only where DynamoDB writes are tested.

---

## Stretch Goal — Observability Dashboard (Option C)

If time allows, add to existing Analytics page:

| Panel | Data source | What it shows |
|---|---|---|
| Workload Distribution | DynamoDB `workload_tier` attributes | Pie: LARGE / MEDIUM / SMALL hospital count |
| Scheduler Queue | In-memory metrics logged to MLflow | Bar: jobs per tier per run |
| Parallelism Efficiency | `parallelism_efficiency` in ForecastsTable | Line chart over time |
| FL Convergence | `FLRoundsTable` | Line: avg_rmse per round |

Backend: one new route `GET /api/distributed-metrics` reading from DynamoDB + MLflow.
Frontend: new tab in `AnalyticsPage.jsx` using existing Recharts components.

---

## File Map

| Action | Path |
|---|---|
| Create | `ml/workload/__init__.py` |
| Create | `ml/workload/profiler.py` |
| Create | `ml/scheduler/__init__.py` |
| Create | `ml/scheduler/priority_queue.py` |
| Create | `ml/parallel/__init__.py` |
| Create | `ml/parallel/executor.py` |
| Create | `ml/federated/__init__.py` |
| Create | `ml/federated/local_trainer.py` |
| Create | `ml/federated/aggregator.py` |
| Create | `ml/federated/round_tracker.py` |
| Create | `tests/test_workload_profiler.py` |
| Create | `tests/test_tier_scheduler.py` |
| Create | `tests/test_parallel_executor.py` |
| Create | `tests/test_federated_aggregator.py` |
| Modify | `lambdas/forecast/handler.py` |
| Modify | `template.yaml` (add FLRoundsTable) |
| Modify | `shared/dynamo.py` (add FL table helpers) |
| Stretch | `backend/app/main.py` (add /api/distributed-metrics) |
| Stretch | `frontend/src/pages/AnalyticsPage.jsx` (add dashboard tab) |
