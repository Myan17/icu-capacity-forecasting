"""Forecast Lambda — Phase 3 core implementation.

Flow:
  1. Load snapshot history for hospital from DynamoDB (up to 52 weeks)
  2. Download model artifacts from S3 (if pre-trained artifacts exist)
     else run BestModelSelector to pick and fit the best model
  3. Generate forecast with confidence intervals (yhat_lower / yhat_upper)
  4. Compute probability-of-breach for each step
  5. Write forecast items to DynamoDB (replacing previous run)
  6. Write probabilistic alerts (deduped by breach window)

Runs inside a Docker container image (ECR) to support Prophet + statsmodels.
Observability: Lambda Powertools structured logging + custom CloudWatch metrics.

Note on serialization: model artifacts are loaded from a private S3 bucket with
server-side encryption and block-public-access enabled (see template.yaml).
The serialization format (joblib/pickle) is appropriate for scikit-learn-compatible
objects; bucket access is restricted to the ForecastFunction IAM role only.
"""
from __future__ import annotations

import io
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import boto3
import joblib          # safer ML serialization than raw pickle
import numpy as np
import pandas as pd
from aws_lambda_powertools import Logger, Metrics, Tracer
from aws_lambda_powertools.metrics import MetricUnit
from boto3.dynamodb.conditions import Key

# ML pipeline (bundled in container image)
import sys
sys.path.insert(0, "/var/task")

from ml.config import DEFAULT_CONFIG
from ml.models.baseline import BaselineForecaster
from ml.models.prophet_model import ProphetForecaster
from ml.models.sarima_model import SarimaForecaster
from ml.model_selection.selector import BestModelSelector

# Spec-conformant distributed-ML modules (L1 → L4):
from ml.workload.profiler import (
    Tier as SpecTier,
    WorkloadProfile as SpecWorkloadProfile,
    WorkloadProfiler as SpecWorkloadProfiler,
)
from ml.scheduler.priority_queue import TierScheduler
from ml.parallel.executor import ParallelExecutor
from ml.federated.round_tracker import RoundTracker
from ml.federated.local_trainer import LocalTrainer
from ml.federated.aggregator import FederatedAggregator

from shared.dynamo import (
    alert_dedupe_key,
    alert_exists,
    alert_sk,
    alerts_table,
    fl_round_pk,
    fl_round_sk,
    fl_rounds_table,
    forecast_sk,
    forecasts_table,
    query_recent_snapshots,
    snapshot_pk,
    snapshots_table,
)
from shared.hospital_tiers import profile_hospital, recommended_resources
from shared.priority_queue import HospitalSignal, PriorityScheduler
from shared.workload_metrics import (
    WorkloadKind,
    WorkloadProfiler,
    detect_bottleneck,
    emit_powertools_metrics,
)
from shared.models import ForecastEvent
from lambdas.forecast.forecast_utils import breach_prob, risk_label, mc_ci

logger  = Logger(service="hospital-forecast")
tracer  = Tracer(service="hospital-forecast")
metrics = Metrics(namespace="HospitalForecasting", service="hospital-forecast")

S3 = boto3.client("s3")
DATA_BUCKET      = os.environ.get("DATA_BUCKET", "")
YELLOW_THRESHOLD = float(os.environ.get("ALERT_OCCUPANCY_YELLOW", 0.75))
RED_THRESHOLD    = float(os.environ.get("ALERT_OCCUPANCY_RED", 0.90))
BREACH_THRESHOLD = float(os.environ.get("BREACH_PROB_THRESHOLD", 0.60))
HORIZON          = int(os.environ.get("FORECAST_HORIZON_WEEKS", 4))
MC_SAMPLES       = 500   # Default; overridden per-hospital by tier

# L2 retry / fallback
MAX_FIT_ATTEMPTS    = int(os.environ.get("MAX_FIT_ATTEMPTS", 2))
ENABLE_FALLBACK     = os.environ.get("ENABLE_FALLBACK", "1") == "1"
ENABLE_BATCHING     = os.environ.get("ENABLE_BATCHING", "1") == "1"
ENABLE_PRIORITY     = os.environ.get("ENABLE_PRIORITY", "1") == "1"
ENABLE_FL           = os.environ.get("ENABLE_FL", "0") == "1"
FORECAST_WORKERS    = max(int(os.environ.get("FORECAST_WORKERS", 4)), 1)


# ── Entry point ───────────────────────────────────────────────────────────────

@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: Any) -> dict:
    """Entry point for the Forecast Lambda.

    Two execution paths:
      * **SQS-triggered** — one record per hospital (BatchSize=1 in
        template.yaml). The per-hospital handler re-raises on failure so SQS
        retries the message (up to maxReceiveCount=3) before routing to DLQ.
      * **Direct invocation** — EventBridge weekly schedule or manual
        ``hospital_ids: [...]`` payload, which fans out through the
        distributed-ML pipeline below.

    Direct-invocation pipeline (Lectures 1 → 4):

      ┌──────────────────────────┐
      │ resolve hospital_ids      │ from event or DynamoDB scan
      └─────────────┬────────────┘
                    ▼
      ┌──────────────────────────┐
      │ WorkloadProfiler         │ L1: classify each hospital (S/M/L)
      └─────────────┬────────────┘
                    ▼
      ┌──────────────────────────┐
      │ TierScheduler            │ L2: priority sort + bin-pack SMALL
      └─────────────┬────────────┘
                    ▼
      ┌──────────────────────────┐
      │ ParallelExecutor         │ L3: ThreadPool over JobBatches
      └─────────────┬────────────┘
                    ▼
      ┌──────────────────────────┐
      │ _run_fl_aggregation      │ L4: FedAvg (ENABLE_FL + all-hospitals path)
      └──────────────────────────┘

    Direct invocations accept ``hospital_ids: [...]`` for explicit batches,
    a single ``hospital_id``, or no key at all (in which case we scan for
    every hospital with snapshots and run the full pipeline plus FL).
    """
    event = event or {}

    if "Records" in event:
        return _handle_sqs_event(event["Records"])

    payload = ForecastEvent(**{k: v for k, v in event.items() if k in {"hospital_id"}})

    explicit_ids = event.get("hospital_ids")
    if explicit_ids:
        hospital_ids = list(explicit_ids)
    else:
        hospital_ids = _resolve_hospital_ids(payload.hospital_id)

    use_priority = ENABLE_PRIORITY and bool(event.get("priority", True))

    profiles, last_forecast_ages = _build_scheduling_profiles(hospital_ids)
    if not use_priority:
        last_forecast_ages = {}

    batches = TierScheduler().schedule(profiles, last_forecast_ages=last_forecast_ages)
    job_results = ParallelExecutor(max_workers=FORECAST_WORKERS).run(
        batches, run_fn=_run_forecast_for_hospital,
    )

    scheduled_ids = {hid for job in job_results for hid in job.hospital_ids}
    results: list[dict] = [r for job in job_results for r in job.results]

    # Hospitals with no snapshots at all (or that failed _build_scheduling_profiles)
    # never reach the executor; surface them as insufficient_data so callers
    # still get one record per requested hospital_id.
    for hid in hospital_ids:
        if hid not in scheduled_ids:
            results.append({"hospital_id": hid, "status": "insufficient_data", "rows": 0})

    parallelism_efficiency = job_results[0].parallelism_efficiency if job_results else 0.0

    if ENABLE_FL and not payload.hospital_id and not explicit_ids:
        try:
            _run_fl_aggregation(hospital_ids)
        except Exception as exc:  # noqa: BLE001 — FL is best-effort
            logger.warning("FL aggregation failed", extra={"error": str(exc)})

    return {
        "status":                 "complete",
        "hospitals":              results,
        "batched":                len(hospital_ids) > 1,
        "priority_used":          use_priority,
        "parallelism_efficiency": parallelism_efficiency,
        "num_batches":            len(batches),
    }


def _handle_sqs_event(records: list[dict]) -> dict:
    """Process SQS records. Raises on failure so SQS retries unprocessed messages."""
    results = []
    for record in records:
        body = json.loads(record["body"])
        hospital_id = body.get("hospital_id")
        if not hospital_id:
            logger.warning("SQS record missing hospital_id — skipping", extra={"body": body})
            continue
        result = _run_forecast_for_hospital(hospital_id)  # raises → SQS retry
        results.append(result)
    return {"status": "complete", "hospitals": results}


def _resolve_hospital_ids(hospital_id: str | None) -> list[str]:
    if hospital_id:
        return [hospital_id]
    table = snapshots_table()
    resp = table.scan(ProjectionExpression="pk", Select="SPECIFIC_ATTRIBUTES")
    pks = {item["pk"] for item in resp.get("Items", [])}
    return [pk.replace("HOSPITAL#", "") for pk in pks if pk.startswith("HOSPITAL#")]


def _prioritise_hospitals(hospital_ids: list[str]) -> list[str]:
    """L2 (legacy): order by current occupancy ratio.

    Kept for backwards compatibility — the new ``TierScheduler`` pipeline now
    handles ordering through the ``last_forecast_ages`` dict produced by
    :func:`_build_scheduling_profiles`.
    """
    sched = PriorityScheduler()
    for hid in hospital_ids:
        items = query_recent_snapshots(hid, limit=8)
        df = _items_to_dataframe(items) if items else pd.DataFrame()
        profile = profile_hospital(hid, df)

        occ_ratio = 0.0
        if not df.empty:
            cap = float(df["icu_capacity"].iloc[-1] or 1)
            occ_ratio = float(df["icu_occupied"].iloc[-1]) / max(cap, 1.0)
        sched.submit(HospitalSignal(profile=profile, occupancy_ratio=occ_ratio))
    return [item.hospital_id for item in sched.drain()]


# ── Spec-conformant scheduling pipeline (Lectures 1 + 2) ──────────────────────

_SPEC_PROFILER = SpecWorkloadProfiler()


def _build_scheduling_profiles(
    hospital_ids: list[str],
) -> tuple[list[SpecWorkloadProfile], dict[str, float]]:
    """Build per-hospital ``WorkloadProfile`` + staleness ages for ``TierScheduler``.

    The "age" we feed the scheduler is the current occupancy ratio: a
    higher ratio means the hospital's last forecast is *more* out of date in
    a risk-weighted sense, so the scheduler runs that hospital first within
    its tier. This preserves the existing "high-occupancy first" behaviour
    while letting the tier ordering dominate (LARGE before MEDIUM before
    SMALL).

    Hospitals that have never reported snapshots are left out of the
    schedule — the caller surfaces them as ``insufficient_data`` afterwards.
    """
    profiles: list[SpecWorkloadProfile] = []
    ages: dict[str, float] = {}
    for hid in hospital_ids:
        try:
            raw_items = query_recent_snapshots(hid, limit=52)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Snapshot query failed; skipping hospital",
                extra={"hospital_id": hid, "error": str(exc)},
            )
            continue
        if not raw_items:
            continue
        df = _items_to_dataframe(raw_items)
        profiles.append(_SPEC_PROFILER.profile(hid, df))

        try:
            cap = float(df["icu_capacity"].iloc[-1] or 1)
            ages[hid] = float(df["icu_occupied"].iloc[-1]) / max(cap, 1.0)
        except Exception:
            ages[hid] = 0.0

    return profiles, ages


def _run_fl_aggregation(hospital_ids: list[str]) -> None:
    """L4: run one FL round across all hospitals (local train + FedAvg).

    Each hospital trains locally, exports a small weight dict, and uploads it
    to S3; the aggregator then averages the dicts and writes the global model
    + a DynamoDB round record. Best-effort — failures are logged and don't
    fail the broader forecast invocation.
    """
    trainer = LocalTrainer(s3_client=S3, data_bucket=DATA_BUCKET)
    aggregator = FederatedAggregator(s3_client=S3, data_bucket=DATA_BUCKET)
    tracker = RoundTracker()

    weight_dicts: list[dict[str, Any]] = []
    for hid in hospital_ids:
        try:
            raw_items = query_recent_snapshots(hid, limit=52)
            if len(raw_items) < 4:
                continue
            df = _items_to_dataframe(raw_items)
            spec_profile = _SPEC_PROFILER.profile(hid, df)
            if not tracker.should_participate(spec_profile.tier):
                continue
            weights = trainer.train_and_extract_weights(
                hid, df, round_n=tracker.current_round,
            )
            weight_dicts.append(weights)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FL local training failed",
                extra={"hospital_id": hid, "error": str(exc)},
            )

    if not weight_dicts:
        logger.info("FL: no eligible hospitals this round; skipping aggregation")
        return

    global_model = aggregator.aggregate(
        round_n=tracker.current_round, weight_dicts=weight_dicts,
    )
    logger.info("FL aggregation complete", extra={"global_model": global_model})

    # Persist a compact round record so we can build a convergence chart later.
    try:
        fl_rounds_table().put_item(Item={
            "pk":                fl_round_pk(),
            "sk":                fl_round_sk(tracker.current_round),
            "round":             int(tracker.current_round),
            "num_clients":       int(global_model.get("num_clients", 0)),
            "total_rows":        int(global_model.get("total_rows", 0)),
            "dominant_model":    str(global_model.get("dominant_model", "")),
            "global_last_value": Decimal(
                str(round(float(global_model.get("global_last_value", 0.0)), 4))
            ),
            "created_at":        datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:  # noqa: BLE001
        logger.warning("FL round record write failed", extra={"error": str(exc)})

    metrics.add_metric(
        name="FLRoundsCompleted", unit=MetricUnit.Count, value=1,
    )


# ── Per-hospital pipeline ─────────────────────────────────────────────────────

@tracer.capture_method
def _run_forecast_for_hospital(hospital_id: str) -> dict:
    """Per-hospital forecast pipeline with workload profiling + retry/fallback.

    Stages:
      load (IO) → select_model (CPU, retry+fallback) → generate (CPU)
      → write_forecasts (IO) → write_alerts (IO)

    Each stage is timed and tagged so a single CloudWatch metric stream tells
    us which stage dominates wall time and whether the workload is CPU- or
    I/O-bound (Lecture 1 / MOS).
    """
    logger.info("Starting forecast", extra={"hospital_id": hospital_id})
    profiler = WorkloadProfiler()

    with profiler.stage("load_snapshots", WorkloadKind.IO):
        raw_items = query_recent_snapshots(hospital_id, limit=52)
    if len(raw_items) < 4:
        return {"hospital_id": hospital_id, "status": "insufficient_data", "rows": len(raw_items)}

    with profiler.stage("frame_construction", WorkloadKind.MIXED):
        df = _items_to_dataframe(raw_items)
        latest_capacity = int(df["icu_capacity"].iloc[-1])

    with profiler.stage("hospital_profile", WorkloadKind.CPU):
        profile = profile_hospital(hospital_id, df)
        budget = recommended_resources(profile)

    with profiler.stage("select_model", WorkloadKind.CPU):
        _LAST_FALLBACK["used"] = False
        model, model_name = _select_model(hospital_id, df)
        fallback_used = bool(_LAST_FALLBACK.get("used", False))

    with profiler.stage("generate_forecast", WorkloadKind.CPU):
        steps = _generate_forecast(model, model_name, df, mc_samples=budget["mc_samples"])

    run_id = uuid.uuid4().hex[:8]
    with profiler.stage("write_forecasts", WorkloadKind.IO):
        _write_forecasts(hospital_id, run_id, model_name, steps, latest_capacity)
    with profiler.stage("write_alerts", WorkloadKind.IO):
        _write_probabilistic_alerts(hospital_id, steps, latest_capacity)

    bottleneck = detect_bottleneck(profiler.report())
    summary = profiler.report().summary()

    metrics.add_metric(name="ForecastsGenerated", unit=MetricUnit.Count, value=1)
    if fallback_used:
        metrics.add_metric(name="ModelFallbackUsed", unit=MetricUnit.Count, value=1)
    metrics.add_metadata(key="workload_tier", value=profile.workload_tier.value)
    metrics.add_metadata(key="bottleneck", value=bottleneck)
    emit_powertools_metrics(profiler.report())

    logger.info(
        "Forecast complete",
        extra={
            "hospital_id":    hospital_id,
            "model":          model_name,
            "run_id":         run_id,
            "fallback_used":  fallback_used,
            "workload_tier":  profile.workload_tier.value,
            "bottleneck":     bottleneck,
            "stage_summary":  summary["wall_by_kind"],
        },
    )
    return {
        "hospital_id":   hospital_id,
        "status":        "success",
        "model":         model_name,
        "run_id":        run_id,
        "workload_tier": profile.workload_tier.value,
        "bottleneck":    bottleneck,
        "fallback_used": fallback_used,
        "wall_ms":       summary["total_wall_ms"],
    }


def _items_to_dataframe(items: list[dict]) -> pd.DataFrame:
    rows = [
        {
            "timestamp":    pd.to_datetime(item["timestamp"]),
            "icu_occupied": float(item.get("icu_occupied", 0)),
            "icu_capacity": float(item.get("icu_capacity", 1)),
            "hospital_id":  item.get("hospital_id", ""),
        }
        for item in items
    ]
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


# ── Model selection ───────────────────────────────────────────────────────────

# Module-level scratch — populated by ``_select_model`` whenever the
# fallback path is taken so the handler can record it as a CloudWatch metric.
_LAST_FALLBACK: dict[str, Any] = {"used": False}


def _select_model(hospital_id: str, df: pd.DataFrame):
    """Public selector entry point.

    Performs L2 retry + fallback to baseline. Returns ``(model, name)`` for
    backward compatibility with existing tests that monkeypatch this function.
    Whether the fallback path was taken is recorded in ``_LAST_FALLBACK``.
    """
    model, name, fallback_used = _select_model_with_fallback(hospital_id, df)
    _LAST_FALLBACK["used"] = fallback_used
    return model, name


def _select_model_with_fallback(
    hospital_id: str, df: pd.DataFrame
) -> tuple[Any, str, bool]:
    """L2 retry-and-fallback. Returns ``(model, name, fallback_used)``.

    Strategy:
      1. Try the S3 artifact first (fast path).
      2. Otherwise, run BestModelSelector with up to ``MAX_FIT_ATTEMPTS`` tries.
      3. If selection still fails and ``ENABLE_FALLBACK`` is on, fall back to
         a Baseline forecaster fit on the raw series. The handler logs the
         fallback and emits a ``ModelFallbackUsed`` CloudWatch metric so we
         can alert on degraded mode.
    """
    artifact = _load_artifact_from_s3(hospital_id)
    if artifact:
        logger.info(
            "Using pre-trained artifact",
            extra={"hospital_id": hospital_id, "model": artifact["model_name"]},
        )
        return artifact["model"], artifact["model_name"], False

    last_exc: Exception | None = None
    for attempt in range(1, max(MAX_FIT_ATTEMPTS, 1) + 1):
        try:
            logger.info(
                "Running BestModelSelector",
                extra={"hospital_id": hospital_id, "attempt": attempt},
            )
            selector = BestModelSelector(
                metric=DEFAULT_CONFIG.selection_metric,
                horizon=min(DEFAULT_CONFIG.test_size, max(len(df) // 4, 2)),
                timestamp_col="timestamp",
                target_col="icu_occupied",
                frequency="W",
            )
            result = selector.select_best_model(df.copy())
            return result.best_model, result.best_model_name, False
        except Exception as exc:  # noqa: BLE001 — we want any failure
            last_exc = exc
            logger.warning(
                "Selector attempt failed",
                extra={"hospital_id": hospital_id, "attempt": attempt, "error": str(exc)},
            )

    if not ENABLE_FALLBACK:
        raise RuntimeError(f"BestModelSelector failed after retries: {last_exc}") from last_exc

    logger.error(
        "All selector attempts failed; falling back to baseline",
        extra={"hospital_id": hospital_id, "error": str(last_exc)},
    )
    fallback = BaselineForecaster(target_col="icu_occupied")
    fallback.fit(df)
    return fallback, "baseline", True


def _load_artifact_from_s3(hospital_id: str):
    """Load a joblib-serialized model artifact from the private data bucket."""
    if not DATA_BUCKET:
        return None
    key = f"artifacts/models/{hospital_id}_best.joblib"
    try:
        obj = S3.get_object(Bucket=DATA_BUCKET, Key=key)
        buf = io.BytesIO(obj["Body"].read())
        return joblib.load(buf)
    except Exception:
        return None


# ── Forecast + intervals ──────────────────────────────────────────────────────

def _generate_forecast(
    model,
    model_name: str,
    df: pd.DataFrame,
    *,
    mc_samples: int | None = None,
) -> list[dict]:
    if isinstance(model, ProphetForecaster):
        return _prophet_forecast(model, df)
    return _mc_forecast(model, model_name, df, mc_samples=mc_samples or MC_SAMPLES)


def _prophet_forecast(model: ProphetForecaster, df: pd.DataFrame) -> list[dict]:
    raw = model.predict(HORIZON)
    now = datetime.now(timezone.utc)
    steps = []
    for i in range(HORIZON):
        if isinstance(raw, pd.DataFrame) and len(raw) > i:
            yhat       = float(raw.iloc[i].get("yhat",       raw.iloc[i].iloc[0]))
            yhat_lower = float(raw.iloc[i].get("yhat_lower", yhat * 0.9))
            yhat_upper = float(raw.iloc[i].get("yhat_upper", yhat * 1.1))
        else:
            val = float(raw[i]) if hasattr(raw, "__getitem__") else float(raw)
            yhat = yhat_lower = yhat_upper = val
        steps.append({
            "forecast_time": now + timedelta(weeks=i + 1),
            "yhat":          yhat,
            "yhat_lower":    yhat_lower,
            "yhat_upper":    yhat_upper,
        })
    return steps


def _mc_forecast(
    model,
    model_name: str,
    df: pd.DataFrame,
    *,
    mc_samples: int = MC_SAMPLES,
) -> list[dict]:
    """Monte Carlo bootstrap for CIs on Baseline/SARIMA.

    ``mc_samples`` is sized per workload tier: smaller hospitals get fewer
    draws so the Lambda finishes within their tier's time budget.
    """
    series = df["icu_occupied"].values.astype(float)
    preds = np.array(model.predict(HORIZON), dtype=float)

    split = max(len(series) - HORIZON, int(len(series) * 0.8))
    train_df = df.iloc[:split].copy()
    actuals = series[split:]
    residuals = _compute_residuals(model_name, train_df, len(actuals), actuals)

    now = datetime.now(timezone.utc)
    steps = []
    for i, yhat in enumerate(preds):
        lo, hi = mc_ci(residuals, float(yhat), seed=42 + i, n_samples=mc_samples)
        steps.append({
            "forecast_time": now + timedelta(weeks=i + 1),
            "yhat":          float(yhat),
            "yhat_lower":    lo,
            "yhat_upper":    hi,
        })
    return steps


def _compute_residuals(model_name: str, train_df: pd.DataFrame, n: int, actuals: np.ndarray) -> np.ndarray:
    try:
        if model_name == "baseline":
            m = BaselineForecaster(target_col="icu_occupied")
        elif model_name == "sarima":
            m = SarimaForecaster(target_col="icu_occupied")
        else:
            m = BaselineForecaster(target_col="icu_occupied")
        m.fit(train_df)
        preds = np.array(m.predict(n), dtype=float)
        k = min(len(preds), len(actuals))
        return actuals[:k] - preds[:k]
    except Exception as exc:
        logger.warning("Residual computation failed", extra={"error": str(exc)})
        return np.zeros(1)


# ── Breach probability ────────────────────────────────────────────────────────

def _breach_prob(step: dict, capacity: int) -> float:
    return breach_prob(
        yhat=step["yhat"],
        yhat_lower=step["yhat_lower"],
        yhat_upper=step["yhat_upper"],
        capacity=capacity,
        red_threshold=RED_THRESHOLD,
    )


# ── DynamoDB writes ───────────────────────────────────────────────────────────

def _write_forecasts(
    hospital_id: str, run_id: str, model_name: str,
    steps: list[dict], capacity: int,
) -> None:
    table = forecasts_table()
    # Clear previous forecasts for this hospital
    pk = snapshot_pk(hospital_id)
    old = table.query(KeyConditionExpression=Key("pk").eq(pk)).get("Items", [])
    with table.batch_writer() as batch:
        for item in old:
            batch.delete_item(Key={"pk": item["pk"], "sk": item["sk"]})

    with table.batch_writer() as batch:
        for step in steps:
            ft: datetime = step["forecast_time"]
            yhat = step["yhat"]
            risk = risk_label(yhat, capacity, yellow=YELLOW_THRESHOLD, red=RED_THRESHOLD)
            bp = _breach_prob(step, capacity)

            batch.put_item(Item={
                "pk":                      snapshot_pk(hospital_id),
                "sk":                      forecast_sk(run_id, ft),
                "hospital_id":             hospital_id,
                "run_id":                  run_id,
                "forecast_time":           ft.isoformat(),
                "predicted_icu_occupied":  Decimal(str(round(yhat, 2))),
                "yhat_lower":              Decimal(str(round(step["yhat_lower"], 2))),
                "yhat_upper":              Decimal(str(round(step["yhat_upper"], 2))),
                "risk_level":              risk,
                "model_name":              model_name,
                "breach_prob":             Decimal(str(round(bp, 4))),
                "icu_capacity":            capacity,
            })


def _write_probabilistic_alerts(hospital_id: str, steps: list[dict], capacity: int) -> None:
    table = alerts_table()
    now = datetime.now(timezone.utc)

    for step in steps:
        bp = _breach_prob(step, capacity)
        if bp < BREACH_THRESHOLD:
            continue

        ft: datetime = step["forecast_time"]
        risk = risk_label(step["yhat"], capacity, yellow=YELLOW_THRESHOLD, red=RED_THRESHOLD)
        eta = round((ft - now).total_seconds() / 3600, 1)
        dedupe = alert_dedupe_key(hospital_id, risk, ft)

        if alert_exists(hospital_id, dedupe):
            continue

        table.put_item(Item={
            "pk":               snapshot_pk(hospital_id),
            "sk":               alert_sk(now),
            "hospital_id":      hospital_id,
            "created_at":       now.isoformat(),
            "risk_level":       risk,
            "message":          (
                f"P(ICU ≥ {int(RED_THRESHOLD*100)}% capacity) = {bp:.0%} "
                f"at {ft.strftime('%Y-%m-%d')} (ETA {eta:.0f}h)"
            ),
            "breach_prob":      Decimal(str(round(bp, 4))),
            "breach_eta_hours": Decimal(str(eta)),
            "dedupe_key":       dedupe,
        })
        metrics.add_metric(name="AlertsRaised", unit=MetricUnit.Count, value=1)
