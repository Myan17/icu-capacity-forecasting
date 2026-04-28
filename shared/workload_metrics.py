"""Workload-aware metrics: CPU vs I/O vs communication time.

Lectures 1 + 3 motivation:
  * MOS expects each pipeline to declare what kind of resource it stresses.
  * Distributed-ML wants explicit communication-overhead numbers so the
    auto-optimizer can decide whether parallelism is worth it.

Two complementary tools:

1. ``WorkloadProfiler`` — context manager + decorator that records
   per-stage wall time, CPU time, peak memory delta, and a tag
   (cpu / io / comm). Emits a structured summary at the end.

2. ``emit_powertools_metrics`` — best-effort bridge that pushes the same
   numbers into AWS Lambda Powertools' ``Metrics`` namespace when running in
   a Lambda. Falls back to a no-op outside Lambda.

The profiler is intentionally pure-Python (stdlib only) so it works in
local benchmarks, in pytest, and inside the Forecast Docker container.
"""
from __future__ import annotations

import json
import logging
import os
import resource
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)


class WorkloadKind(str, Enum):
    CPU = "cpu"      # Heavy numerical work (model fit, predict)
    IO = "io"        # Disk / network / DynamoDB / S3
    COMM = "comm"    # Inter-stage data transfer (parquet round-trips)
    MIXED = "mixed"  # Anything else


@dataclass
class StageTiming:
    name: str
    kind: WorkloadKind
    wall_ms: float
    cpu_ms: float
    rss_delta_kb: int
    payload_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


@dataclass
class WorkloadReport:
    stages: list[StageTiming] = field(default_factory=list)

    def add(self, stage: StageTiming) -> None:
        self.stages.append(stage)

    def total_ms(self) -> float:
        return sum(s.wall_ms for s in self.stages)

    def summary(self) -> dict[str, Any]:
        by_kind: dict[str, float] = {}
        for s in self.stages:
            by_kind[s.kind.value] = by_kind.get(s.kind.value, 0.0) + s.wall_ms
        total = self.total_ms() or 1.0
        ratios = {k: round(v / total, 3) for k, v in by_kind.items()}
        return {
            "stages":         [s.to_dict() for s in self.stages],
            "total_wall_ms":  round(total, 2),
            "wall_by_kind":   {k: round(v, 2) for k, v in by_kind.items()},
            "ratio_by_kind":  ratios,
            "dominant_kind":  max(by_kind, key=by_kind.get) if by_kind else "unknown",
        }

    def to_json(self) -> str:
        return json.dumps(self.summary(), default=str)


class WorkloadProfiler:
    """Per-pipeline profiler used by the forecast handler and ML pipelines.

    Usage::

        prof = WorkloadProfiler()
        with prof.stage("load_snapshots", WorkloadKind.IO):
            df = load_from_dynamo(...)

        with prof.stage("fit_model", WorkloadKind.CPU):
            model.fit(df)

        report = prof.report()
        emit_powertools_metrics(report, namespace="HospitalForecasting")
    """

    def __init__(self) -> None:
        self._report = WorkloadReport()

    @contextmanager
    def stage(
        self,
        name: str,
        kind: WorkloadKind = WorkloadKind.MIXED,
        payload_bytes: int = 0,
    ) -> Iterator[None]:
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        rss_start = _peak_rss_kb()
        try:
            yield
        finally:
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
            cpu_ms = (time.process_time() - cpu_start) * 1000.0
            rss_delta = _peak_rss_kb() - rss_start
            self._report.add(
                StageTiming(
                    name=name,
                    kind=kind,
                    wall_ms=round(wall_ms, 3),
                    cpu_ms=round(cpu_ms, 3),
                    rss_delta_kb=int(rss_delta),
                    payload_bytes=int(payload_bytes),
                )
            )

    def report(self) -> WorkloadReport:
        return self._report


def _peak_rss_kb() -> int:
    """Resident-set-size delta. macOS reports bytes, Linux reports KB.
    We normalize to KB."""
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return 0
    if rss > 10**8:  # bytes (macOS)
        return rss // 1024
    return int(rss)


def detect_bottleneck(report: WorkloadReport) -> str:
    """Return a one-word bottleneck classification.

    Rules:
      cpu > 60% wall  → "cpu_bound"
      io  > 60% wall  → "io_bound"
      comm > 30% wall → "comm_bound"
      else            → "balanced"
    """
    summary = report.summary()
    ratios = summary.get("ratio_by_kind", {})
    if ratios.get("cpu", 0) > 0.6:
        return "cpu_bound"
    if ratios.get("io", 0) > 0.6:
        return "io_bound"
    if ratios.get("comm", 0) > 0.3:
        return "comm_bound"
    return "balanced"


def emit_powertools_metrics(report: WorkloadReport, namespace: str = "HospitalForecasting") -> None:
    """Best-effort: push wall_by_kind into Lambda Powertools Metrics.

    Silently no-ops if powertools isn't installed (e.g. local pytest)."""
    try:
        from aws_lambda_powertools import Metrics
        from aws_lambda_powertools.metrics import MetricUnit
    except Exception:
        return

    metrics = Metrics(namespace=namespace, service="hospital-forecast")
    summary = report.summary()
    for kind, ms in summary["wall_by_kind"].items():
        metrics.add_metric(
            name=f"WallMs_{kind.capitalize()}",
            unit=MetricUnit.Milliseconds,
            value=float(ms),
        )
    metrics.add_metric(
        name="WallMs_Total",
        unit=MetricUnit.Milliseconds,
        value=float(summary["total_wall_ms"]),
    )
    metrics.add_metadata(key="dominant_kind", value=summary["dominant_kind"])


def time_function(kind: WorkloadKind = WorkloadKind.MIXED) -> Callable:
    """Decorator that prints (and returns) timing info for a single call.

    Light-weight alternative to the WorkloadProfiler context manager; use
    when you only need to time one function."""

    def deco(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            c0 = time.process_time()
            out = func(*args, **kwargs)
            wall = (time.perf_counter() - t0) * 1000.0
            cpu = (time.process_time() - c0) * 1000.0
            logger.info(
                "fn_timing",
                extra={
                    "fn": func.__name__,
                    "kind": kind.value,
                    "wall_ms": round(wall, 3),
                    "cpu_ms": round(cpu, 3),
                },
            )
            if os.environ.get("WORKLOAD_TRACE") == "1":
                print(f"[{kind.value:>5}] {func.__name__:<32} wall={wall:7.2f}ms cpu={cpu:7.2f}ms")
            return out

        wrapper.__wrapped__ = func
        return wrapper

    return deco


__all__ = [
    "WorkloadKind",
    "StageTiming",
    "WorkloadReport",
    "WorkloadProfiler",
    "detect_bottleneck",
    "emit_powertools_metrics",
    "time_function",
]
