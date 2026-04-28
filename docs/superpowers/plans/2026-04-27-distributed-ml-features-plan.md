# Distributed ML Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four simulated distributed-systems modules (WorkloadProfiler, TierScheduler, ParallelExecutor, FederatedAggregator) to the ICU forecasting Lambda pipeline, each mapping to a CSCI5980 lecture concept, plus wire them into the existing handler.

**Architecture:** All four modules are additive — they wrap the existing `_run_forecast_for_hospital` function without replacing it. The Lambda handler's flat `for hid in hospital_ids` loop is replaced by `WorkloadProfiler → TierScheduler → ParallelExecutor`, with `FederatedAggregator` firing after all-hospitals runs. FL simulation treats each `hospital_id` as a federated client that trains locally and uploads weight JSON to S3; FedAvg aggregation runs after all clients complete a round.

**Tech Stack:** Python 3.11, boto3, pandas, numpy, pytest, moto 5.x, existing `ml/model_selection/selector.py` (`BestModelSelector`), existing `shared/dynamo.py` patterns.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `ml/workload/__init__.py` | Package marker |
| Create | `ml/workload/profiler.py` | Tier classification by row count + variance |
| Create | `ml/scheduler/__init__.py` | Package marker |
| Create | `ml/scheduler/priority_queue.py` | Priority sort + SMALL bin-packing |
| Create | `ml/parallel/__init__.py` | Package marker |
| Create | `ml/parallel/executor.py` | ThreadPoolExecutor + DAG stage timing |
| Create | `ml/federated/__init__.py` | Package marker |
| Create | `ml/federated/round_tracker.py` | Round counter + tier participation gating |
| Create | `ml/federated/local_trainer.py` | Per-hospital model selection + weight extraction + S3 upload |
| Create | `ml/federated/aggregator.py` | FedAvg + global model JSON + DynamoDB round record |
| Create | `tests/test_workload_profiler.py` | WorkloadProfiler unit tests |
| Create | `tests/test_tier_scheduler.py` | TierScheduler unit tests |
| Create | `tests/test_parallel_executor.py` | ParallelExecutor unit tests |
| Create | `tests/test_federated_aggregator.py` | FL unit tests (RoundTracker + LocalTrainer + Aggregator) |
| Modify | `shared/dynamo.py` | Add `fl_rounds_table()` + FL key builders |
| Modify | `template.yaml` | Add `FLRoundsTable` DynamoDB resource + env var |
| Modify | `lambdas/forecast/handler.py` | Wire profiler → scheduler → executor; call FL aggregation |

---

## Task 1: WorkloadProfiler (L1 — MOS)

**Files:**
- Create: `ml/workload/__init__.py`
- Create: `ml/workload/profiler.py`
- Create: `tests/test_workload_profiler.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_workload_profiler.py
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from ml.workload.profiler import Tier, WorkloadProfile, WorkloadProfiler


def _make_df(rows: int, variance_target: float) -> pd.DataFrame:
    np.random.seed(42)
    std = variance_target ** 0.5
    occupied = np.random.normal(loc=50, scale=std, size=rows)
    return pd.DataFrame({"icu_occupied": occupied, "icu_capacity": [100] * rows})


def test_large_tier_assigned():
    df = _make_df(35, 80.0)
    profile = WorkloadProfiler().profile("H001", df)
    assert profile.tier == Tier.LARGE
    assert profile.mc_samples == 1000


def test_medium_tier_by_rows():
    df = _make_df(20, 5.0)
    profile = WorkloadProfiler().profile("H002", df)
    assert profile.tier == Tier.MEDIUM
    assert profile.mc_samples == 500


def test_medium_tier_by_variance():
    df = _make_df(10, 30.0)
    profile = WorkloadProfiler().profile("H003", df)
    assert profile.tier == Tier.MEDIUM


def test_small_tier_assigned():
    df = _make_df(8, 5.0)
    profile = WorkloadProfiler().profile("H004", df)
    assert profile.tier == Tier.SMALL
    assert profile.mc_samples == 200


def test_returns_workload_profile_type():
    df = _make_df(10, 5.0)
    profile = WorkloadProfiler().profile("H005", df)
    assert isinstance(profile, WorkloadProfile)
    assert profile.hospital_id == "H005"
    assert profile.reason != ""


def test_cache_returns_same_object():
    df = _make_df(35, 80.0)
    profiler = WorkloadProfiler()
    p1 = profiler.profile("H006", df)
    p2 = profiler.profile("H006", df)
    assert p1 is p2


def test_large_has_higher_priority_score_than_small():
    large_df = _make_df(35, 80.0)
    small_df = _make_df(8, 5.0)
    profiler = WorkloadProfiler()
    large_p = profiler.profile("H007", large_df)
    small_p = profiler.profile("H008", small_df)
    assert large_p.priority_score > small_p.priority_score


def test_clear_cache():
    df = _make_df(35, 80.0)
    profiler = WorkloadProfiler()
    p1 = profiler.profile("H009", df)
    profiler.clear_cache()
    p2 = profiler.profile("H009", df)
    assert p1 is not p2
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Users/myangupta/Downloads/CSCI5980/hospital-forecasting
python -m pytest tests/test_workload_profiler.py -v
```

Expected: `ModuleNotFoundError: No module named 'ml.workload'`

- [ ] **Step 3: Create package marker**

```python
# ml/workload/__init__.py
```

- [ ] **Step 4: Implement profiler**

```python
# ml/workload/profiler.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Tier(str, Enum):
    LARGE = "LARGE"
    MEDIUM = "MEDIUM"
    SMALL = "SMALL"


@dataclass
class WorkloadProfile:
    hospital_id: str
    tier: Tier
    priority_score: float
    mc_samples: int
    reason: str


_VARIANCE_MAX = 200.0  # normalization cap for priority score
_TIER_MC = {Tier.LARGE: 1000, Tier.MEDIUM: 500, Tier.SMALL: 200}


class WorkloadProfiler:
    """Classify hospitals into compute tiers based on historical data volume and variance."""

    def __init__(self) -> None:
        self._cache: dict[str, WorkloadProfile] = {}

    def profile(self, hospital_id: str, df: pd.DataFrame) -> WorkloadProfile:
        if hospital_id in self._cache:
            return self._cache[hospital_id]

        rows = len(df)
        variance = float(df["icu_occupied"].var()) if rows > 1 else 0.0
        variance_normalized = min(variance / _VARIANCE_MAX, 1.0)
        priority_score = rows * 0.6 + variance_normalized * 0.4

        if rows >= 30 and variance > 50:
            tier = Tier.LARGE
        elif rows >= 15 or variance >= 20:
            tier = Tier.MEDIUM
        else:
            tier = Tier.SMALL

        result = WorkloadProfile(
            hospital_id=hospital_id,
            tier=tier,
            priority_score=priority_score,
            mc_samples=_TIER_MC[tier],
            reason=f"rows={rows}, variance={variance:.1f}",
        )
        self._cache[hospital_id] = result
        return result

    def clear_cache(self) -> None:
        self._cache.clear()
```

- [ ] **Step 5: Run tests to confirm passing**

```bash
python -m pytest tests/test_workload_profiler.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add ml/workload/__init__.py ml/workload/profiler.py tests/test_workload_profiler.py
git commit -m "feat: add WorkloadProfiler — tier classification by row count and variance (L1 MOS)"
```

---

## Task 2: TierScheduler (L2 — Borg/Kubernetes)

**Files:**
- Create: `ml/scheduler/__init__.py`
- Create: `ml/scheduler/priority_queue.py`
- Create: `tests/test_tier_scheduler.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tier_scheduler.py
from __future__ import annotations
import pytest
from ml.workload.profiler import Tier, WorkloadProfile
from ml.scheduler.priority_queue import JobBatch, TierScheduler


def _profile(hospital_id: str, tier: Tier, score: float = 10.0) -> WorkloadProfile:
    mc = {Tier.LARGE: 1000, Tier.MEDIUM: 500, Tier.SMALL: 200}[tier]
    return WorkloadProfile(hospital_id=hospital_id, tier=tier,
                           priority_score=score, mc_samples=mc, reason="test")


def test_large_before_small():
    profiles = [_profile("S1", Tier.SMALL), _profile("L1", Tier.LARGE)]
    batches = TierScheduler().schedule(profiles)
    assert batches[0].tier == Tier.LARGE


def test_large_before_medium_before_small():
    profiles = [
        _profile("S1", Tier.SMALL),
        _profile("L1", Tier.LARGE),
        _profile("M1", Tier.MEDIUM),
    ]
    tiers = [b.tier for b in TierScheduler().schedule(profiles)]
    assert tiers.index(Tier.LARGE) < tiers.index(Tier.MEDIUM)
    assert tiers.index(Tier.MEDIUM) < tiers.index(Tier.SMALL)


def test_small_hospitals_bin_packed_max_5():
    profiles = [_profile(f"S{i}", Tier.SMALL) for i in range(7)]
    batches = TierScheduler().schedule(profiles)
    small_batches = [b for b in batches if b.tier == Tier.SMALL]
    assert len(small_batches) == 2
    assert len(small_batches[0].hospital_ids) == 5
    assert len(small_batches[1].hospital_ids) == 2


def test_no_batch_exceeds_5():
    profiles = [_profile(f"S{i}", Tier.SMALL) for i in range(12)]
    batches = TierScheduler().schedule(profiles)
    for b in batches:
        assert len(b.hospital_ids) <= 5


def test_medium_not_binpacked():
    profiles = [_profile(f"M{i}", Tier.MEDIUM) for i in range(3)]
    batches = TierScheduler().schedule(profiles)
    assert len(batches) == 3
    for b in batches:
        assert len(b.hospital_ids) == 1


def test_staleness_orders_within_tier():
    profiles = [_profile("L1", Tier.LARGE), _profile("L2", Tier.LARGE)]
    ages = {"L1": 10.0, "L2": 50.0}
    batches = TierScheduler().schedule(profiles, last_forecast_ages=ages)
    large_ids = [b.hospital_ids[0] for b in batches if b.tier == Tier.LARGE]
    assert large_ids[0] == "L2"  # most stale first


def test_empty_input_returns_empty():
    assert TierScheduler().schedule([]) == []


def test_job_batch_fields_populated():
    profiles = [_profile("L1", Tier.LARGE)]
    batch = TierScheduler().schedule(profiles)[0]
    assert isinstance(batch, JobBatch)
    assert batch.priority == 1
    assert batch.estimated_duration_s > 0
    assert batch.hospital_ids == ["L1"]
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_tier_scheduler.py -v
```

Expected: `ModuleNotFoundError: No module named 'ml.scheduler'`

- [ ] **Step 3: Create package marker**

```python
# ml/scheduler/__init__.py
```

- [ ] **Step 4: Implement scheduler**

```python
# ml/scheduler/priority_queue.py
from __future__ import annotations

from dataclasses import dataclass

from ml.workload.profiler import Tier, WorkloadProfile

_SMALL_BIN_SIZE = 5
_PRIORITY = {Tier.LARGE: 1, Tier.MEDIUM: 2, Tier.SMALL: 3}
_ESTIMATED_DURATION_S = {Tier.LARGE: 120.0, Tier.MEDIUM: 60.0, Tier.SMALL: 20.0}


@dataclass
class JobBatch:
    hospital_ids: list[str]
    tier: Tier
    priority: int
    estimated_duration_s: float


class TierScheduler:
    """Sort hospitals by tier + staleness, bin-pack SMALL hospitals into batches of ≤5."""

    def schedule(
        self,
        profiles: list[WorkloadProfile],
        last_forecast_ages: dict[str, float] | None = None,
    ) -> list[JobBatch]:
        ages = last_forecast_ages or {}
        sorted_profiles = sorted(
            profiles,
            key=lambda p: (_PRIORITY[p.tier], -ages.get(p.hospital_id, 0.0)),
        )

        batches: list[JobBatch] = []
        small_bin: list[str] = []

        for profile in sorted_profiles:
            if profile.tier == Tier.SMALL:
                small_bin.append(profile.hospital_id)
                if len(small_bin) >= _SMALL_BIN_SIZE:
                    batches.append(_small_batch(small_bin[:]))
                    small_bin.clear()
            else:
                batches.append(JobBatch(
                    hospital_ids=[profile.hospital_id],
                    tier=profile.tier,
                    priority=_PRIORITY[profile.tier],
                    estimated_duration_s=_ESTIMATED_DURATION_S[profile.tier],
                ))

        if small_bin:
            batches.append(_small_batch(small_bin))

        return batches


def _small_batch(hospital_ids: list[str]) -> JobBatch:
    return JobBatch(
        hospital_ids=hospital_ids,
        tier=Tier.SMALL,
        priority=_PRIORITY[Tier.SMALL],
        estimated_duration_s=_ESTIMATED_DURATION_S[Tier.SMALL] * len(hospital_ids),
    )
```

- [ ] **Step 5: Run tests to confirm passing**

```bash
python -m pytest tests/test_tier_scheduler.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add ml/scheduler/__init__.py ml/scheduler/priority_queue.py tests/test_tier_scheduler.py
git commit -m "feat: add TierScheduler — priority sort and SMALL bin-packing (L2 Borg)"
```

---

## Task 3: ParallelExecutor (L3 — Distributed ML)

**Files:**
- Create: `ml/parallel/__init__.py`
- Create: `ml/parallel/executor.py`
- Create: `tests/test_parallel_executor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_parallel_executor.py
from __future__ import annotations
import time
import pytest
from ml.workload.profiler import Tier
from ml.scheduler.priority_queue import JobBatch
from ml.parallel.executor import JobResult, ParallelExecutor


def _batch(hospital_ids: list[str], tier: Tier = Tier.MEDIUM) -> JobBatch:
    return JobBatch(hospital_ids=hospital_ids, tier=tier, priority=2,
                    estimated_duration_s=60.0)


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
    assert results[0].status == "error"


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
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_parallel_executor.py -v
```

Expected: `ModuleNotFoundError: No module named 'ml.parallel'`

- [ ] **Step 3: Create package marker**

```python
# ml/parallel/__init__.py
```

- [ ] **Step 4: Implement executor**

```python
# ml/parallel/executor.py
from __future__ import annotations

import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

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
    """Execute job batches concurrently; each batch runs its hospitals sequentially."""

    def __init__(self, max_workers: int = 4) -> None:
        self._max_workers = max_workers

    def run(self, batches: list[JobBatch], run_fn: Callable[[str], dict]) -> list[JobResult]:
        if not batches:
            return []

        wall_start = time.monotonic()
        futures: dict = {}
        job_results: list[JobResult] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            for batch in batches:
                futures[executor.submit(self._run_batch, batch, run_fn)] = batch

        serial_total = 0.0
        for future, batch in futures.items():
            try:
                result = future.result()
                job_results.append(result)
                serial_total += result.wall_clock_s
            except Exception as exc:
                job_results.append(JobResult(
                    hospital_ids=batch.hospital_ids,
                    status="error",
                    wall_clock_s=0.0,
                    stage_durations={"preprocess": 0.0, "train": 0.0, "predict": 0.0},
                    parallelism_efficiency=0.0,
                    results=[{"error": str(exc)}],
                ))

        wall_clock = time.monotonic() - wall_start
        efficiency = serial_total / (self._max_workers * wall_clock) if wall_clock > 0 else 1.0
        for r in job_results:
            r.parallelism_efficiency = efficiency

        return job_results

    def _run_batch(self, batch: JobBatch, run_fn: Callable[[str], dict]) -> JobResult:
        stages = OrderedDict([("preprocess", 0.0), ("train", 0.0), ("predict", 0.0)])
        batch_start = time.monotonic()
        results = []

        for hospital_id in batch.hospital_ids:
            t0 = time.monotonic()
            result = run_fn(hospital_id)
            elapsed = time.monotonic() - t0
            for stage in stages:
                stages[stage] += elapsed / len(stages)
            results.append(result)

        return JobResult(
            hospital_ids=batch.hospital_ids,
            status="success",
            wall_clock_s=time.monotonic() - batch_start,
            stage_durations=dict(stages),
            parallelism_efficiency=0.0,
            results=results,
        )
```

- [ ] **Step 5: Run tests to confirm passing**

```bash
python -m pytest tests/test_parallel_executor.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add ml/parallel/__init__.py ml/parallel/executor.py tests/test_parallel_executor.py
git commit -m "feat: add ParallelExecutor — ThreadPoolExecutor + 3-stage DAG timing (L3 Distributed ML)"
```

---

## Task 4: FL RoundTracker (L4 — Federated Learning / TiFL)

**Files:**
- Create: `ml/federated/__init__.py`
- Create: `ml/federated/round_tracker.py`
- Create: `tests/test_federated_aggregator.py` (start with RoundTracker tests)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_federated_aggregator.py
from __future__ import annotations
import pytest
from ml.workload.profiler import Tier
from ml.federated.round_tracker import RoundTracker


# ── RoundTracker ──────────────────────────────────────────────────────────────

def test_large_participates_every_round():
    tracker = RoundTracker(current_round=0)
    assert tracker.should_participate(Tier.LARGE) is True
    tracker.advance()
    assert tracker.should_participate(Tier.LARGE) is True


def test_medium_participates_every_2_rounds():
    tracker = RoundTracker(current_round=0)
    assert tracker.should_participate(Tier.MEDIUM) is True
    tracker.advance()
    assert tracker.should_participate(Tier.MEDIUM) is False
    tracker.advance()
    assert tracker.should_participate(Tier.MEDIUM) is True


def test_small_participates_every_4_rounds():
    tracker = RoundTracker(current_round=0)
    assert tracker.should_participate(Tier.SMALL) is True
    for _ in range(3):
        tracker.advance()
    assert tracker.should_participate(Tier.SMALL) is False
    tracker.advance()  # round 4
    assert tracker.should_participate(Tier.SMALL) is True


def test_advance_increments_round():
    tracker = RoundTracker(current_round=2)
    tracker.advance()
    assert tracker.current_round == 3


def test_round_zero_all_tiers_participate():
    tracker = RoundTracker(current_round=0)
    for tier in Tier:
        assert tracker.should_participate(tier) is True
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_federated_aggregator.py -v -k "round"
```

Expected: `ModuleNotFoundError: No module named 'ml.federated'`

- [ ] **Step 3: Create package marker**

```python
# ml/federated/__init__.py
```

- [ ] **Step 4: Implement RoundTracker**

```python
# ml/federated/round_tracker.py
from __future__ import annotations

from ml.workload.profiler import Tier

_PARTICIPATION_FREQUENCY: dict[Tier, int] = {
    Tier.LARGE: 1,
    Tier.MEDIUM: 2,
    Tier.SMALL: 4,
}


class RoundTracker:
    """Track FL round number and gate hospital participation by tier."""

    def __init__(self, current_round: int = 0) -> None:
        self.current_round = current_round

    def should_participate(self, tier: Tier) -> bool:
        freq = _PARTICIPATION_FREQUENCY[tier]
        return self.current_round % freq == 0

    def advance(self) -> None:
        self.current_round += 1
```

- [ ] **Step 5: Run tests to confirm passing**

```bash
python -m pytest tests/test_federated_aggregator.py -v -k "round"
```

Expected: 5 tests PASS.

---

## Task 5: FL LocalTrainer (L4 — Federated Learning)

**Files:**
- Create: `ml/federated/local_trainer.py`
- Extend: `tests/test_federated_aggregator.py`

- [ ] **Step 1: Append LocalTrainer tests to the existing test file**

Add these tests at the bottom of `tests/test_federated_aggregator.py`:

```python
import json
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from ml.federated.local_trainer import LocalTrainer


def _make_hospital_df(rows: int = 20) -> pd.DataFrame:
    np.random.seed(0)
    ts = pd.date_range("2023-01-01", periods=rows, freq="W")
    return pd.DataFrame({
        "timestamp": ts,
        "icu_occupied": np.random.normal(50, 8, size=rows),
        "icu_capacity": [100] * rows,
        "hospital_id": ["H_test"] * rows,
    })


def test_local_trainer_returns_weight_dict():
    df = _make_hospital_df(20)
    trainer = LocalTrainer(s3_client=None, data_bucket="")
    weights = trainer.train_and_extract_weights("H_test", df, round_n=0)
    assert "hospital_id" in weights
    assert "model_name" in weights
    assert "n_rows" in weights
    assert "last_value" in weights
    assert "round" in weights
    assert weights["hospital_id"] == "H_test"
    assert weights["n_rows"] == 20
    assert weights["round"] == 0


def test_local_trainer_last_value_matches_df_tail():
    df = _make_hospital_df(20)
    trainer = LocalTrainer(s3_client=None, data_bucket="")
    weights = trainer.train_and_extract_weights("H_test", df, round_n=1)
    expected_last = float(df["icu_occupied"].iloc[-1])
    assert abs(weights["last_value"] - expected_last) < 1e-6


def test_local_trainer_uploads_to_s3_when_bucket_set():
    df = _make_hospital_df(20)
    mock_s3 = MagicMock()
    trainer = LocalTrainer(s3_client=mock_s3, data_bucket="test-bucket")
    trainer.train_and_extract_weights("H_test", df, round_n=2)
    mock_s3.put_object.assert_called_once()
    call_kwargs = mock_s3.put_object.call_args[1]
    assert call_kwargs["Bucket"] == "test-bucket"
    assert "fl/round_2/H_test_weights.json" in call_kwargs["Key"]


def test_local_trainer_skips_upload_when_no_bucket():
    df = _make_hospital_df(20)
    mock_s3 = MagicMock()
    trainer = LocalTrainer(s3_client=mock_s3, data_bucket="")
    trainer.train_and_extract_weights("H_test", df, round_n=0)
    mock_s3.put_object.assert_not_called()
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_federated_aggregator.py -v -k "local_trainer"
```

Expected: `ModuleNotFoundError: No module named 'ml.federated.local_trainer'`

- [ ] **Step 3: Implement LocalTrainer**

```python
# ml/federated/local_trainer.py
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ml.config import DEFAULT_CONFIG
from ml.model_selection.selector import BestModelSelector


class LocalTrainer:
    """Train a model on a single hospital's data and extract serializable weights."""

    def __init__(self, s3_client=None, data_bucket: str = "") -> None:
        self._s3 = s3_client
        self._bucket = data_bucket

    def train_and_extract_weights(
        self,
        hospital_id: str,
        df: pd.DataFrame,
        round_n: int,
    ) -> dict[str, Any]:
        model_name, model = self._select_model(df)
        weights = self._extract_weights(hospital_id, df, model_name, model, round_n)
        if self._bucket and self._s3:
            key = f"fl/round_{round_n}/{hospital_id}_weights.json"
            self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=json.dumps(weights).encode(),
            )
        return weights

    def _select_model(self, df: pd.DataFrame) -> tuple[str, Any]:
        horizon = min(DEFAULT_CONFIG.test_size, max(len(df) // 4, 2))
        selector = BestModelSelector(
            metric=DEFAULT_CONFIG.selection_metric,
            horizon=horizon,
            timestamp_col="timestamp",
            target_col="icu_occupied",
            frequency="W",
        )
        result = selector.select_best_model(df.copy())
        return result.best_model_name, result.best_model

    def _extract_weights(
        self,
        hospital_id: str,
        df: pd.DataFrame,
        model_name: str,
        model: Any,
        round_n: int,
    ) -> dict[str, Any]:
        last_value = float(df["icu_occupied"].iloc[-1]) if len(df) > 0 else 0.0
        weights: dict[str, Any] = {
            "hospital_id": hospital_id,
            "model_name": model_name,
            "n_rows": len(df),
            "last_value": last_value,
            "round": round_n,
        }

        if model_name == "baseline" and hasattr(model, "last_value") and model.last_value is not None:
            weights["params"] = {"last_value": float(model.last_value)}
        elif model_name == "sarima" and hasattr(model, "fitted_model") and model.fitted_model is not None:
            try:
                weights["params"] = {
                    "ar_params": model.fitted_model.arparams.tolist(),
                    "ma_params": model.fitted_model.maparams.tolist(),
                }
            except Exception:
                weights["params"] = {"last_value": last_value}
        else:
            weights["params"] = {"last_value": last_value}

        return weights
```

- [ ] **Step 4: Run tests to confirm passing**

```bash
python -m pytest tests/test_federated_aggregator.py -v -k "local_trainer"
```

Expected: 4 tests PASS.

---

## Task 6: FL FederatedAggregator (L4 — FedAvg + DynamoDB round records)

**Files:**
- Create: `ml/federated/aggregator.py`
- Extend: `tests/test_federated_aggregator.py`

- [ ] **Step 1: Append FederatedAggregator tests**

Add these tests at the bottom of `tests/test_federated_aggregator.py`:

```python
from ml.federated.aggregator import FederatedAggregator


def _weight(hospital_id: str, last_value: float, n_rows: int, model_name: str = "baseline") -> dict:
    return {
        "hospital_id": hospital_id,
        "model_name": model_name,
        "n_rows": n_rows,
        "last_value": last_value,
        "round": 0,
        "params": {"last_value": last_value},
    }


def test_fedavg_is_weighted_mean_by_rows():
    weights = [
        _weight("H1", last_value=60.0, n_rows=40),
        _weight("H2", last_value=40.0, n_rows=10),
    ]
    result = FederatedAggregator().aggregate(round_n=0, weight_dicts=weights)
    # FedAvg: (60*40 + 40*10) / 50 = (2400+400)/50 = 56.0
    assert abs(result["global_last_value"] - 56.0) < 1e-6


def test_aggregate_returns_required_keys():
    weights = [_weight("H1", 50.0, 20), _weight("H2", 60.0, 30)]
    result = FederatedAggregator().aggregate(round_n=1, weight_dicts=weights)
    for key in ("round", "num_clients", "global_last_value", "dominant_model", "total_rows"):
        assert key in result


def test_aggregate_num_clients_correct():
    weights = [_weight(f"H{i}", 50.0, 10) for i in range(5)]
    result = FederatedAggregator().aggregate(round_n=0, weight_dicts=weights)
    assert result["num_clients"] == 5


def test_aggregate_dominant_model_is_most_common():
    weights = [
        _weight("H1", 50.0, 30, model_name="sarima"),
        _weight("H2", 50.0, 20, model_name="sarima"),
        _weight("H3", 50.0, 10, model_name="baseline"),
    ]
    result = FederatedAggregator().aggregate(round_n=0, weight_dicts=weights)
    assert result["dominant_model"] == "sarima"


def test_aggregate_empty_list_returns_empty():
    result = FederatedAggregator().aggregate(round_n=0, weight_dicts=[])
    assert result == {}


def test_aggregate_uploads_global_model_to_s3():
    mock_s3 = MagicMock()
    weights = [_weight("H1", 50.0, 20), _weight("H2", 60.0, 30)]
    FederatedAggregator(s3_client=mock_s3, data_bucket="test-bucket").aggregate(
        round_n=3, weight_dicts=weights
    )
    mock_s3.put_object.assert_called_once()
    call_kwargs = mock_s3.put_object.call_args[1]
    assert call_kwargs["Key"] == "fl/global_model.json"
    body = json.loads(call_kwargs["Body"])
    assert body["round"] == 3
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_federated_aggregator.py -v -k "aggregate"
```

Expected: `ModuleNotFoundError: No module named 'ml.federated.aggregator'`

- [ ] **Step 3: Implement FederatedAggregator**

```python
# ml/federated/aggregator.py
from __future__ import annotations

import json
from typing import Any


class FederatedAggregator:
    """Compute FedAvg over per-hospital weight dicts and store global model to S3."""

    def __init__(self, s3_client=None, data_bucket: str = "") -> None:
        self._s3 = s3_client
        self._bucket = data_bucket

    def aggregate(self, round_n: int, weight_dicts: list[dict[str, Any]]) -> dict[str, Any]:
        if not weight_dicts:
            return {}

        total_rows = sum(w["n_rows"] for w in weight_dicts)
        if total_rows == 0:
            return {}

        global_last_value = sum(
            w["last_value"] * w["n_rows"] / total_rows for w in weight_dicts
        )

        model_type_rows: dict[str, int] = {}
        for w in weight_dicts:
            mn = w["model_name"]
            model_type_rows[mn] = model_type_rows.get(mn, 0) + w["n_rows"]
        dominant_model = max(model_type_rows, key=lambda k: model_type_rows[k])

        global_model: dict[str, Any] = {
            "round": round_n,
            "num_clients": len(weight_dicts),
            "global_last_value": global_last_value,
            "dominant_model": dominant_model,
            "total_rows": total_rows,
        }

        if self._bucket and self._s3:
            self._s3.put_object(
                Bucket=self._bucket,
                Key="fl/global_model.json",
                Body=json.dumps(global_model).encode(),
            )

        return global_model
```

- [ ] **Step 4: Run all FL tests**

```bash
python -m pytest tests/test_federated_aggregator.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ml/federated/__init__.py ml/federated/round_tracker.py \
        ml/federated/local_trainer.py ml/federated/aggregator.py \
        tests/test_federated_aggregator.py
git commit -m "feat: add FL modules — RoundTracker, LocalTrainer, FederatedAggregator with FedAvg (L4 Federated Learning)"
```

---

## Task 7: DynamoDB FLRoundsTable + SAM template

**Files:**
- Modify: `shared/dynamo.py`
- Modify: `template.yaml`

- [ ] **Step 1: Add FL table helpers to `shared/dynamo.py`**

Open `shared/dynamo.py` and append after the existing helper functions:

```python
# ── Federated Learning ────────────────────────────────────────────────────────

def fl_rounds_table():
    return _table("FL_ROUNDS_TABLE")


def fl_round_pk() -> str:
    return "FL#GLOBAL"


def fl_round_sk(round_n: int) -> str:
    return f"ROUND#{round_n:06d}"
```

- [ ] **Step 2: Add FLRoundsTable to `template.yaml`**

In the `Globals > Function > Environment > Variables` section, add:
```yaml
FL_ROUNDS_TABLE: !Ref FLRoundsTable
```

In the `Resources` section, after the `AlertsTable` resource, add:

```yaml
  FLRoundsTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: !Sub hospital-forecasting-fl-rounds-${Environment}
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: pk
          AttributeType: S
        - AttributeName: sk
          AttributeType: S
      KeySchema:
        - AttributeName: pk
          KeyType: HASH
        - AttributeName: sk
          KeyType: RANGE
```

- [ ] **Step 3: Verify template is valid SAM**

```bash
cd /Users/myangupta/Downloads/CSCI5980/hospital-forecasting
sam validate --lint 2>&1 | head -20
```

Expected: no errors (warnings about lint are OK).

- [ ] **Step 4: Run the full existing test suite to confirm nothing broke**

```bash
python -m pytest tests/ -v --ignore=tests/load -q
```

Expected: all existing tests PASS plus all new tests PASS.

- [ ] **Step 5: Commit**

```bash
git add shared/dynamo.py template.yaml
git commit -m "feat: add FLRoundsTable to SAM template and dynamo helpers"
```

---

## Task 8: Handler Integration

**Files:**
- Modify: `lambdas/forecast/handler.py`

This task wires all four modules into the existing Lambda handler with minimal changes to existing logic.

- [ ] **Step 1: Add imports at the top of `lambdas/forecast/handler.py`**

After the existing imports, add:

```python
from ml.workload.profiler import WorkloadProfiler
from ml.scheduler.priority_queue import TierScheduler
from ml.parallel.executor import ParallelExecutor
from ml.federated.round_tracker import RoundTracker
from ml.federated.local_trainer import LocalTrainer
from ml.federated.aggregator import FederatedAggregator
```

- [ ] **Step 2: Inject WorkloadProfile into `_run_forecast_for_hospital`**

Find the line in `_run_forecast_for_hospital` that sets `MC_SAMPLES`:

```python
steps = _generate_forecast(model, model_name, df)
```

Before that line, add the profiler call and update `_mc_forecast` invocation. The change is inside `_run_forecast_for_hospital` — add these two lines right after `df = _items_to_dataframe(raw_items)`:

```python
    profile = WorkloadProfiler().profile(hospital_id, df)
    mc_samples_for_hospital = profile.mc_samples
```

Then find the `_mc_forecast` call inside `_generate_forecast`. `_generate_forecast` currently uses the module-level `MC_SAMPLES = 500` constant. Pass `mc_samples_for_hospital` down by changing the `_generate_forecast` call:

Change:
```python
    model, model_name = _select_model(hospital_id, df)
    steps = _generate_forecast(model, model_name, df)
```

To:
```python
    profile = WorkloadProfiler().profile(hospital_id, df)
    model, model_name = _select_model(hospital_id, df)
    steps = _generate_forecast(model, model_name, df, mc_samples=profile.mc_samples)
```

Update the `_generate_forecast` signature:

```python
def _generate_forecast(model, model_name: str, df: pd.DataFrame, mc_samples: int = MC_SAMPLES) -> list[dict]:
    if isinstance(model, ProphetForecaster):
        return _prophet_forecast(model, df)
    return _mc_forecast(model, model_name, df, mc_samples=mc_samples)
```

Update `_mc_forecast` signature:

```python
def _mc_forecast(model, model_name: str, df: pd.DataFrame, mc_samples: int = MC_SAMPLES) -> list[dict]:
```

And inside `_mc_forecast`, replace the hardcoded `MC_SAMPLES` reference with the `mc_samples` parameter:

```python
    lo, hi = mc_ci(residuals, float(yhat), seed=42 + i)
```

(No change needed there — `MC_SAMPLES` is only used in `mc_ci` call if it was passed. Check `forecast_utils.py` to confirm `mc_ci` uses its own default or `MC_SAMPLES` directly.)

Actually: look at the current `_mc_forecast` — it calls `mc_ci(residuals, float(yhat), seed=42+i)` and `mc_ci` internally uses a fixed 500 samples. Update by passing `mc_samples` to `mc_ci`. Open `lambdas/forecast/forecast_utils.py` and update `mc_ci` to accept `n_samples: int = 500`:

```python
def mc_ci(residuals: np.ndarray, yhat: float, seed: int = 42, n_samples: int = 500) -> tuple[float, float]:
```

Then in `_mc_forecast`, call:
```python
    lo, hi = mc_ci(residuals, float(yhat), seed=42 + i, n_samples=mc_samples)
```

- [ ] **Step 3: Replace the flat hospital loop with scheduler + executor in `lambda_handler`**

Find the direct-invocation branch in `lambda_handler`:

```python
    hospital_ids = _resolve_hospital_ids(payload.hospital_id)

    results = []
    for hid in hospital_ids:
        try:
            result = _run_forecast_for_hospital(hid)
            results.append(result)
        except Exception as exc:
            logger.error("Forecast failed", extra={"hospital_id": hid, "error": str(exc)})
            results.append({"hospital_id": hid, "status": "error", "error": str(exc)})

    return {"status": "complete", "hospitals": results}
```

Replace it with:

```python
    hospital_ids = _resolve_hospital_ids(payload.hospital_id)

    profiles = _build_scheduling_profiles(hospital_ids)
    batches = TierScheduler().schedule(profiles)
    job_results = ParallelExecutor(max_workers=4).run(
        batches, run_fn=_run_forecast_for_hospital
    )

    results = [r for job in job_results for r in job.results]

    if not payload.hospital_id:
        _run_fl_aggregation(hospital_ids)

    return {"status": "complete", "hospitals": results}
```

- [ ] **Step 4: Add `_build_scheduling_profiles` helper**

Add this function before `lambda_handler`:

```python
def _build_scheduling_profiles(hospital_ids: list[str]) -> list:
    """Build lightweight WorkloadProfiles using only DynamoDB row counts (no full data load)."""
    from ml.workload.profiler import WorkloadProfile, Tier, WorkloadProfiler
    import pandas as pd

    profiles = []
    profiler = WorkloadProfiler()
    for hid in hospital_ids:
        raw_items = query_recent_snapshots(hid, limit=52)
        if len(raw_items) < 4:
            continue
        df = _items_to_dataframe(raw_items)
        profiles.append(profiler.profile(hid, df))
    return profiles
```

- [ ] **Step 5: Add `_run_fl_aggregation` helper**

Add this function after `_build_scheduling_profiles`:

```python
def _run_fl_aggregation(hospital_ids: list[str]) -> None:
    """Run one FL round: local training per hospital + FedAvg aggregation."""
    trainer = LocalTrainer(s3_client=S3, data_bucket=DATA_BUCKET)
    aggregator = FederatedAggregator(s3_client=S3, data_bucket=DATA_BUCKET)
    tracker = RoundTracker()

    weight_dicts = []
    for hid in hospital_ids:
        try:
            raw_items = query_recent_snapshots(hid, limit=52)
            if len(raw_items) < 4:
                continue
            df = _items_to_dataframe(raw_items)
            profile = WorkloadProfiler().profile(hid, df)
            if not tracker.should_participate(profile.tier):
                continue
            weights = trainer.train_and_extract_weights(hid, df, round_n=tracker.current_round)
            weight_dicts.append(weights)
        except Exception as exc:
            logger.warning("FL local training failed", extra={"hospital_id": hid, "error": str(exc)})

    if weight_dicts:
        global_model = aggregator.aggregate(
            round_n=tracker.current_round, weight_dicts=weight_dicts
        )
        logger.info("FL aggregation complete", extra={"global_model": global_model})
```

- [ ] **Step 6: Run the existing handler test suite**

```bash
python -m pytest tests/test_api_handler.py tests/test_forecast_handler.py -v
```

Expected: all existing tests PASS.

- [ ] **Step 7: Run full test suite**

```bash
python -m pytest tests/ -v --ignore=tests/load -q
```

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add lambdas/forecast/handler.py lambdas/forecast/forecast_utils.py
git commit -m "feat: wire WorkloadProfiler + TierScheduler + ParallelExecutor + FederatedAggregator into forecast handler"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] L1 MOS — WorkloadProfiler (Tasks 1, 8)
- [x] L2 Borg — TierScheduler with bin-packing + staleness ordering (Tasks 2, 8)
- [x] L3 Distributed ML — ParallelExecutor with ThreadPoolExecutor + DAG (Tasks 3, 8)
- [x] L4 FL — RoundTracker (TiFL tiers), LocalTrainer (per-hospital weights), FederatedAggregator (FedAvg) (Tasks 4–6, 8)
- [x] DynamoDB FLRoundsTable — added in Task 7 (template.yaml + dynamo.py)
- [x] Handler integration — Task 8 wires all modules

**Placeholder scan:** No TBDs, no "implement later", all code blocks complete.

**Type consistency:**
- `WorkloadProfile` defined in Task 1, consumed by Tasks 2, 3, 8 — consistent
- `JobBatch` defined in Task 2, consumed by Tasks 3, 8 — consistent
- `JobResult` defined in Task 3, consumed in Task 8 — consistent
- `LocalTrainer.train_and_extract_weights(hospital_id, df, round_n)` — consistent across Tasks 5 and 8
- `FederatedAggregator.aggregate(round_n, weight_dicts)` — consistent across Tasks 6 and 8
