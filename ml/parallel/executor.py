"""ParallelExecutor — ThreadPoolExecutor over JobBatches with a 3-stage DAG (L3).

Spec: docs/superpowers/specs/2026-04-27-distributed-ml-features-design.md

Each ``JobBatch`` is dispatched to a worker thread; hospitals within a batch
run sequentially in that thread. The "DAG" is a simple ordered set of three
stages — ``preprocess → train → predict`` — and per-stage durations are
attributed proportionally to the per-hospital wall time.

Reported metrics
----------------
* ``wall_clock_s`` — total elapsed time for the batch.
* ``stage_durations`` — split of wall time across the three pipeline stages.
* ``parallelism_efficiency`` — ``Σ serial_times / (max_workers * wall_clock)``;
  a value < 1 means the threads were not saturated; ≥ 1 means good utilisation.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from ml.scheduler.priority_queue import JobBatch


@dataclass
class JobResult:
    hospital_ids: list[str]
    status: str
    wall_clock_s: float
    stage_durations: dict[str, float]
    parallelism_efficiency: float
    results: list[dict]


class ParallelExecutor:
    """Run job batches concurrently with a fixed-size thread pool."""

    def __init__(self, max_workers: int = 4) -> None:
        self._max_workers = max(int(max_workers), 1)

    def run(
        self,
        batches: list[JobBatch],
        run_fn: Callable[[str], dict],
    ) -> list[JobResult]:
        if not batches:
            return []

        wall_start = time.monotonic()
        job_results: list[JobResult] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures: dict[Any, JobBatch] = {
                executor.submit(self._run_batch, batch, run_fn): batch
                for batch in batches
            }
            for future, batch in futures.items():
                try:
                    job_results.append(future.result())
                except Exception as exc:  # noqa: BLE001 — surface any failure
                    job_results.append(JobResult(
                        hospital_ids=list(batch.hospital_ids),
                        status="error",
                        wall_clock_s=0.0,
                        stage_durations={"preprocess": 0.0, "train": 0.0, "predict": 0.0},
                        parallelism_efficiency=0.0,
                        results=[{
                            "hospital_id": hid,
                            "status": "error",
                            "error": str(exc),
                        } for hid in batch.hospital_ids],
                    ))

        wall_clock = max(time.monotonic() - wall_start, 1e-9)
        serial_total = sum(r.wall_clock_s for r in job_results)
        efficiency = serial_total / (self._max_workers * wall_clock)
        for r in job_results:
            r.parallelism_efficiency = efficiency

        return job_results

    def _run_batch(
        self,
        batch: JobBatch,
        run_fn: Callable[[str], dict],
    ) -> JobResult:
        stages = OrderedDict([("preprocess", 0.0), ("train", 0.0), ("predict", 0.0)])
        batch_start = time.monotonic()
        results: list[dict] = []

        for hospital_id in batch.hospital_ids:
            t0 = time.monotonic()
            try:
                results.append(run_fn(hospital_id))
            except Exception as exc:  # noqa: BLE001
                results.append({
                    "hospital_id": hospital_id,
                    "status": "error",
                    "error": str(exc),
                })
            elapsed = time.monotonic() - t0
            share = elapsed / len(stages)
            for stage in stages:
                stages[stage] += share

        return JobResult(
            hospital_ids=list(batch.hospital_ids),
            status="success",
            wall_clock_s=time.monotonic() - batch_start,
            stage_durations=dict(stages),
            parallelism_efficiency=0.0,
            results=results,
        )


__all__ = ["JobResult", "ParallelExecutor"]
