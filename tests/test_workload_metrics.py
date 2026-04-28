"""Unit tests for shared/workload_metrics.py (L1 + L3 instrumentation)."""
from __future__ import annotations

import time

from shared.workload_metrics import (
    WorkloadKind,
    WorkloadProfiler,
    WorkloadReport,
    detect_bottleneck,
    time_function,
)


def test_profiler_records_stage_timings():
    prof = WorkloadProfiler()
    with prof.stage("io1", WorkloadKind.IO):
        time.sleep(0.01)
    with prof.stage("cpu1", WorkloadKind.CPU):
        sum(i * i for i in range(50_000))
    report = prof.report()
    assert len(report.stages) == 2
    assert report.total_ms() > 0
    summary = report.summary()
    assert summary["dominant_kind"] in {"io", "cpu"}
    assert "io" in summary["wall_by_kind"]
    assert "cpu" in summary["wall_by_kind"]


def test_detect_bottleneck_classifies_correctly():
    prof = WorkloadProfiler()
    with prof.stage("io_block", WorkloadKind.IO):
        time.sleep(0.05)
    with prof.stage("cpu_quick", WorkloadKind.CPU):
        pass
    assert detect_bottleneck(prof.report()) in {"io_bound", "balanced"}


def test_detect_bottleneck_balanced_when_empty():
    assert detect_bottleneck(WorkloadReport()) == "balanced"


def test_time_function_decorator_passes_through_value():
    @time_function(WorkloadKind.CPU)
    def add(a, b):
        return a + b
    assert add(2, 3) == 5
    # Decorator should keep the original function reachable for tests.
    assert add.__wrapped__(2, 3) == 5
