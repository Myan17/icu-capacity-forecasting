"""ParallelExecutor unit tests (spec: docs/superpowers/specs/2026-04-27).

Confirms:
  * All hospitals across all batches are processed.
  * Two workers truly parallelise non-trivial work (wall < N * single).
  * Errors are captured, not propagated.
  * ``stage_durations`` carries the three pipeline-stage keys.
  * Bin-packed batches surface every hospital_id in their results.
"""
from __future__ import annotations

import time

from ml.parallel.executor import JobResult, ParallelExecutor
from ml.scheduler.priority_queue import JobBatch
from ml.workload.profiler import Tier


def _batch(hospital_ids: list[str], tier: Tier = Tier.MEDIUM) -> JobBatch:
    return JobBatch(
        hospital_ids=hospital_ids, tier=tier, priority=2,
        estimated_duration_s=60.0,
    )


def _fast(hospital_id: str) -> dict:
    return {"hospital_id": hospital_id, "status": "success"}


def _slow(hospital_id: str) -> dict:
    time.sleep(0.05)
    return {"hospital_id": hospital_id, "status": "success"}


def _fail(hospital_id: str) -> dict:
    raise ValueError("simulated failure")


def test_all_hospitals_processed():
    batches = [_batch(["H1"]), _batch(["H2"]), _batch(["H3"])]
    results = ParallelExecutor(max_workers=3).run(batches, _fast)
    ids = {hid for r in results for hid in r.hospital_ids}
    assert ids == {"H1", "H2", "H3"}


def test_parallel_faster_than_serial():
    batches = [_batch([f"H{i}"]) for i in range(4)]
    start = time.monotonic()
    ParallelExecutor(max_workers=4).run(batches, _slow)
    wall = time.monotonic() - start
    assert wall < 4 * 0.05 * 0.9  # at least 10% faster than serial


def test_returns_job_result_type():
    results = ParallelExecutor(max_workers=1).run([_batch(["H1"])], _fast)
    assert isinstance(results[0], JobResult)
    assert results[0].status == "success"


def test_error_returns_error_status():
    results = ParallelExecutor(max_workers=1).run([_batch(["H_fail"])], _fail)
    # Per-hospital error captured inside the batch's results list (success status
    # at the batch level since the executor itself didn't crash).
    assert results[0].results[0]["status"] == "error"


def test_empty_batches_returns_empty():
    assert ParallelExecutor(max_workers=2).run([], _fast) == []


def test_stage_durations_has_correct_keys():
    results = ParallelExecutor(max_workers=1).run([_batch(["H1"])], _fast)
    assert set(results[0].stage_durations.keys()) == {"preprocess", "train", "predict"}


def test_parallelism_efficiency_is_positive():
    batches = [_batch(["H1"]), _batch(["H2"])]
    results = ParallelExecutor(max_workers=2).run(batches, _slow)
    for r in results:
        assert r.parallelism_efficiency > 0


def test_binpacked_batch_processes_all_ids():
    batches = [_batch(["H1", "H2", "H3"], tier=Tier.SMALL)]
    results = ParallelExecutor(max_workers=1).run(batches, _fast)
    assert set(results[0].hospital_ids) == {"H1", "H2", "H3"}
    assert len(results[0].results) == 3
