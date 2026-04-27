"""MLflow event hook for Locust — logs per-hospital metrics after each run.

Called automatically by locustfile.py's quitting event. Each call creates
one MLflow run in the 'hospital-load-test' experiment with:
  - Per-hospital metrics: {HOSPITAL_ID}_p50_ms, _p95_ms, _p99_ms,
                          _failure_rate, _rps
  - Summary metrics:      overall_p95_ms, overall_failure_rate, overall_rps
  - Run tags:             num_hospitals, run_duration_s, host, timestamp
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone

import mlflow


EXPERIMENT_NAME = "hospital-load-test"
_HOSPITAL_RE = re.compile(r"/(\d{6})\b")


def _extract_hospital(name: str) -> str | None:
    m = _HOSPITAL_RE.search(name)
    return m.group(1) if m else None


def log_run(environment) -> None:
    """Aggregate Locust stats and write one MLflow run."""
    mlflow.set_experiment(EXPERIMENT_NAME)

    stats = environment.stats
    runner = environment.runner
    host = environment.host or "unknown"

    hospital_buckets: dict[str, list] = {}
    for (name, _method), entry in stats.entries.items():
        hid = _extract_hospital(name)
        if hid is None:
            continue
        hospital_buckets.setdefault(hid, []).append(entry)

    if not hospital_buckets:
        print("[mlflow_reporter] No hospital stats collected — skipping MLflow log.")
        return

    run_duration = round(time.time() - environment.stats.start_time, 1) if environment.stats.start_time else 0

    try:
        with mlflow.start_run():
            p95_values = []
            failure_rates = []
            rps_values = []

            for hid, entries in sorted(hospital_buckets.items()):
                total_reqs = sum(e.num_requests for e in entries)
                total_fails = sum(e.num_failures for e in entries)
                failure_rate = round(total_fails / max(total_reqs, 1), 4)
                rps = round(sum(e.total_rps for e in entries), 3)

                p50 = round(
                    sum((e.get_response_time_percentile(0.50) or 0) * e.num_requests for e in entries)
                    / max(total_reqs, 1), 1
                )
                p95 = round(
                    sum((e.get_response_time_percentile(0.95) or 0) * e.num_requests for e in entries)
                    / max(total_reqs, 1), 1
                )
                p99 = round(
                    sum((e.get_response_time_percentile(0.99) or 0) * e.num_requests for e in entries)
                    / max(total_reqs, 1), 1
                )

                mlflow.log_metrics({
                    f"{hid}_p50_ms":       p50,
                    f"{hid}_p95_ms":       p95,
                    f"{hid}_p99_ms":       p99,
                    f"{hid}_failure_rate": failure_rate,
                    f"{hid}_rps":          rps,
                })

                p95_values.append((p95, total_reqs))
                failure_rates.append((failure_rate, total_reqs))
                rps_values.append(rps)

            if p95_values:
                total_weight = sum(reqs for _, reqs in p95_values) or 1
                mlflow.log_metrics({
                    "overall_p95_ms":       round(sum(p * r for p, r in p95_values) / total_weight, 1),
                    "overall_failure_rate": round(sum(f * r for f, r in failure_rates) / total_weight, 4),
                    "overall_rps":          round(sum(rps_values), 3),
                })

            mlflow.set_tags({
                "num_hospitals":   str(len(hospital_buckets)),
                "run_duration_s":  str(run_duration),
                "host":            host,
                "timestamp":       datetime.now(timezone.utc).isoformat(),
            })

        print(f"[mlflow_reporter] Run logged to experiment '{EXPERIMENT_NAME}'")
    except Exception as exc:
        print(f"[mlflow_reporter] WARNING: failed to log run — {exc}")
