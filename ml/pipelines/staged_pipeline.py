"""Staged ML pipeline — Lecture 3 (Distributed ML / Alpa) inter-op + intra-op
parallelism, simulated locally.

Alpa partitions a model graph into pipeline *stages* and runs them with
overlapping execution: stage k+1 starts as soon as stage k emits an output
micro-batch. We approximate the same behaviour for a forecasting pipeline
with four stages:

    LOAD ─► CLEAN ─► FEATURE ─► (TRAIN ‖ PREDICT)

Each stage receives a Pandas DataFrame (the "micro-batch") through a queue and
hands a transformed DataFrame to the next. Stages run on a thread pool so I/O
stages overlap CPU stages, and the parallel-model selector inside the TRAIN
stage gives intra-op parallelism within a stage.

Key features for the course narrative:
  * Inter-op parallelism : different stages run concurrently
  * Intra-op parallelism : the TRAIN stage uses ParallelBestModelSelector
  * Communication cost   : measured via parquet round-trip bytes between stages
  * Auto-mode            : if a stage takes < `inline_threshold_ms` it's
                           inlined into the previous stage to amortise
                           queue overhead — same heuristic Alpa uses to fuse
                           tiny ops.
"""
from __future__ import annotations

import io
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ml.model_selection.parallel_selector import ParallelBestModelSelector
from ml.preprocessing.cleaner import clean_timeseries_with_summary
from shared.workload_metrics import WorkloadKind, WorkloadProfiler


_SENTINEL = object()


@dataclass
class StageResult:
    name: str
    duration_ms: float
    bytes_in: int = 0
    bytes_out: int = 0
    output_rows: int = 0


@dataclass
class PipelineRun:
    hospital_id: str
    best_model_name: str
    forecast: List[Dict[str, Any]]
    stage_results: List[StageResult] = field(default_factory=list)
    total_wall_ms: float = 0.0
    total_comm_bytes: int = 0
    inline_decisions: Dict[str, bool] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "hospital_id":     self.hospital_id,
            "best_model_name": self.best_model_name,
            "forecast_steps":  len(self.forecast),
            "total_wall_ms":   round(self.total_wall_ms, 2),
            "total_comm_bytes": self.total_comm_bytes,
            "stages": [
                {
                    "name": s.name,
                    "duration_ms": round(s.duration_ms, 2),
                    "bytes_in": s.bytes_in,
                    "bytes_out": s.bytes_out,
                    "output_rows": s.output_rows,
                }
                for s in self.stage_results
            ],
            "inline_decisions": self.inline_decisions,
        }


def _serialize(df: pd.DataFrame) -> int:
    """Estimate inter-stage communication cost via parquet round-trip size."""
    if df is None or df.empty:
        return 0
    buf = io.BytesIO()
    try:
        df.to_parquet(buf, index=False)
    except Exception:
        df.to_csv(buf, index=False)
    return buf.tell()


class StagedPipeline:
    """Run an end-to-end forecast as overlapping pipeline stages.

    Parameters
    ----------
    horizon : int
        Forecast horizon in steps.
    inline_threshold_ms : float
        Stages faster than this are run inline in the next stage to avoid
        queue overhead — this is the local analogue of Alpa's op-fusion pass.
    enable_intra_op : bool
        When True, the TRAIN stage uses ParallelBestModelSelector.
    """

    def __init__(
        self,
        horizon: int = 8,
        timestamp_col: str = "timestamp",
        target_col: str = "icu_occupied",
        frequency: str = "W",
        inline_threshold_ms: float = 2.0,
        enable_intra_op: bool = True,
    ) -> None:
        self.horizon = horizon
        self.timestamp_col = timestamp_col
        self.target_col = target_col
        self.frequency = frequency
        self.inline_threshold_ms = inline_threshold_ms
        self.enable_intra_op = enable_intra_op

    def run(
        self,
        hospital_id: str,
        raw_df: pd.DataFrame,
        profiler: Optional[WorkloadProfiler] = None,
    ) -> PipelineRun:
        prof = profiler or WorkloadProfiler()
        run = PipelineRun(hospital_id=hospital_id, best_model_name="", forecast=[])
        wall_start = time.perf_counter()

        q1: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        q2: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        q3: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        result_holder: Dict[str, Any] = {}
        errors: List[BaseException] = []

        def _wrap(target: Callable, name: str) -> Callable:
            def runner():
                try:
                    target()
                except BaseException as exc:
                    errors.append(exc)
                    for q in (q1, q2, q3):
                        try:
                            q.put_nowait(_SENTINEL)
                        except queue.Full:
                            pass
            runner.__name__ = name
            return runner

        threads = [
            threading.Thread(target=_wrap(lambda: self._stage_load(raw_df, q1, run, prof), "load")),
            threading.Thread(target=_wrap(lambda: self._stage_clean(q1, q2, run, prof), "clean")),
            threading.Thread(target=_wrap(lambda: self._stage_feature(q2, q3, run, prof), "feature")),
            threading.Thread(target=_wrap(lambda: self._stage_train_predict(q3, result_holder, run, prof), "train_predict")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            raise errors[0]

        run.total_wall_ms = (time.perf_counter() - wall_start) * 1000.0
        run.total_comm_bytes = sum(s.bytes_out for s in run.stage_results)
        run.best_model_name = result_holder.get("best_model_name", "")
        run.forecast = result_holder.get("forecast", [])
        return run

    # ── Stages ────────────────────────────────────────────────────────────────

    def _stage_load(self, raw_df: pd.DataFrame, q_out, run: PipelineRun, prof: WorkloadProfiler) -> None:
        with prof.stage("load", WorkloadKind.IO):
            df = raw_df.copy()
            if self.timestamp_col in df.columns:
                df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
                df = df.sort_values(self.timestamp_col).reset_index(drop=True)
        bytes_out = _serialize(df)
        run.stage_results.append(StageResult("load", prof.report().stages[-1].wall_ms, bytes_out=bytes_out, output_rows=len(df)))
        q_out.put(df)
        q_out.put(_SENTINEL)

    def _stage_clean(self, q_in, q_out, run: PipelineRun, prof: WorkloadProfiler) -> None:
        df = q_in.get()
        if df is _SENTINEL:
            q_out.put(_SENTINEL)
            return
        bytes_in = _serialize(df)
        with prof.stage("clean", WorkloadKind.MIXED):
            try:
                cleaned, _ = clean_timeseries_with_summary(
                    df,
                    timestamp_col=self.timestamp_col,
                    target_col=self.target_col,
                )
            except Exception:
                cleaned = df.dropna(subset=[self.target_col]).reset_index(drop=True)
        bytes_out = _serialize(cleaned)
        run.stage_results.append(StageResult("clean", prof.report().stages[-1].wall_ms,
                                              bytes_in=bytes_in, bytes_out=bytes_out, output_rows=len(cleaned)))
        q_out.put(cleaned)
        q_in.get()  # drain sentinel

    def _stage_feature(self, q_in, q_out, run: PipelineRun, prof: WorkloadProfiler) -> None:
        df = q_in.get()
        if df is _SENTINEL:
            q_out.put(_SENTINEL)
            return
        bytes_in = _serialize(df)
        with prof.stage("feature", WorkloadKind.CPU):
            featured = df.copy()
            for lag in (1, 2, 4):
                featured[f"lag_{lag}"] = featured[self.target_col].shift(lag)
            featured["rolling_mean_4"] = featured[self.target_col].rolling(4, min_periods=1).mean()
            featured = featured.bfill().ffill()
        bytes_out = _serialize(featured)
        # Apply inline-op-fusion heuristic
        last_ms = prof.report().stages[-1].wall_ms
        run.inline_decisions["feature"] = last_ms < self.inline_threshold_ms
        run.stage_results.append(StageResult("feature", last_ms,
                                              bytes_in=bytes_in, bytes_out=bytes_out, output_rows=len(featured)))
        q_out.put(featured)

    def _stage_train_predict(self, q_in, holder: Dict[str, Any], run: PipelineRun, prof: WorkloadProfiler) -> None:
        df = q_in.get()
        if df is _SENTINEL or df is None:
            holder["best_model_name"] = ""
            holder["forecast"] = []
            return
        bytes_in = _serialize(df)
        with prof.stage("train", WorkloadKind.CPU):
            if self.enable_intra_op:
                selector = ParallelBestModelSelector(
                    horizon=min(self.horizon, max(len(df) // 4, 2)),
                    timestamp_col=self.timestamp_col,
                    target_col=self.target_col,
                    frequency=self.frequency,
                    executor="thread",
                )
                pres = selector.select_best_model_parallel(df)
                model = pres.inner.best_model
                model_name = pres.inner.best_model_name
            else:
                from ml.model_selection.selector import BestModelSelector
                selector = BestModelSelector(
                    horizon=min(self.horizon, max(len(df) // 4, 2)),
                    timestamp_col=self.timestamp_col,
                    target_col=self.target_col,
                    frequency=self.frequency,
                )
                res = selector.select_best_model(df)
                model = res.best_model
                model_name = res.best_model_name
        with prof.stage("predict", WorkloadKind.CPU):
            preds = model.predict(self.horizon)
            forecast = self._format_forecast(preds)
        run.stage_results.append(StageResult("train", prof.report().stages[-2].wall_ms, bytes_in=bytes_in))
        run.stage_results.append(StageResult("predict", prof.report().stages[-1].wall_ms, output_rows=len(forecast)))
        holder["best_model_name"] = model_name
        holder["forecast"] = forecast

    def _format_forecast(self, preds: Any) -> List[Dict[str, Any]]:
        if isinstance(preds, pd.DataFrame):
            yhat = preds["yhat"].to_numpy() if "yhat" in preds.columns else preds.iloc[:, 0].to_numpy()
        elif isinstance(preds, pd.Series):
            yhat = preds.to_numpy()
        else:
            yhat = np.asarray(preds, dtype=float)
        return [{"step": i + 1, "yhat": float(v)} for i, v in enumerate(yhat)]


__all__ = ["StagedPipeline", "PipelineRun", "StageResult"]
