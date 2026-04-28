"""Multi-process / GPU-ready inference path (Lecture 3).

The Forecast Lambda is the obvious place to scale up inference:
  * Many hospitals to predict for in one invocation (batch forecasting)
  * Each prediction is independent and CPU-bound
  * Process-level parallelism sidesteps the GIL completely

This module provides a small wrapper around ``multiprocessing.Pool`` (CPU)
and a ``cuda``-flag-aware path that delegates to PyTorch when available.
The PyTorch path only kicks in if ``import torch`` succeeds and CUDA is
visible — otherwise we silently fall back to CPU multiprocessing.

For the course report we don't need real GPUs; the important thing is the
*interface* is GPU-ready and the auto-optimizer can pick it.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd


def _gpu_available() -> bool:
    if os.environ.get("FORCE_CPU_INFERENCE") == "1":
        return False
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _predict_one(args: tuple[str, Any, int]) -> Dict[str, Any]:
    hospital_id, model, horizon = args
    t0 = time.perf_counter()
    preds = model.predict(horizon)
    if isinstance(preds, pd.DataFrame):
        arr = preds["yhat"].to_numpy() if "yhat" in preds.columns else preds.iloc[:, 0].to_numpy()
    elif isinstance(preds, pd.Series):
        arr = preds.to_numpy()
    else:
        arr = np.asarray(preds, dtype=float)
    return {
        "hospital_id": hospital_id,
        "predictions": arr.tolist(),
        "duration_ms": (time.perf_counter() - t0) * 1000.0,
    }


@dataclass
class BatchInferenceResult:
    backend: str
    num_workers: int
    per_hospital: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_wall_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend":       self.backend,
            "num_workers":   self.num_workers,
            "total_wall_ms": round(self.total_wall_ms, 2),
            "per_hospital": {
                hid: {**v, "duration_ms": round(v["duration_ms"], 2)}
                for hid, v in self.per_hospital.items()
            },
        }


def predict_batch(
    models: Dict[str, Any],
    horizon: int = 8,
    *,
    backend: str = "auto",
    num_workers: int | None = None,
) -> BatchInferenceResult:
    """Run inference for many hospitals in parallel.

    Parameters
    ----------
    models : dict
        ``{hospital_id: trained_model}``.
    backend : "auto" | "cpu" | "gpu"
        ``auto`` picks gpu if torch+cuda is available, else cpu.
    """
    if backend == "auto":
        backend = "gpu" if _gpu_available() else "cpu"
    if backend == "gpu":
        return _predict_batch_gpu(models, horizon)
    return _predict_batch_cpu(models, horizon, num_workers=num_workers)


def _predict_batch_cpu(
    models: Dict[str, Any],
    horizon: int,
    num_workers: int | None = None,
) -> BatchInferenceResult:
    n_workers = num_workers or min(len(models), os.cpu_count() or 2)
    res = BatchInferenceResult(backend="cpu", num_workers=n_workers)

    t0 = time.perf_counter()
    if n_workers <= 1 or len(models) == 1:
        for hid, model in models.items():
            res.per_hospital[hid] = _predict_one((hid, model, horizon))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_predict_one, (hid, m, horizon)): hid for hid, m in models.items()}
            for fut in as_completed(futures):
                hid = futures[fut]
                try:
                    res.per_hospital[hid] = fut.result()
                except Exception as exc:
                    res.per_hospital[hid] = {"hospital_id": hid, "error": str(exc)}
    res.total_wall_ms = (time.perf_counter() - t0) * 1000.0
    return res


def _predict_batch_gpu(models: Dict[str, Any], horizon: int) -> BatchInferenceResult:
    """Best-effort GPU path. Right now Prophet/SARIMA/Baseline don't have GPU
    ops natively, so we fall through to CPU and just label the backend.

    The plumbing is in place so a future PyTorch-based forecaster (e.g.
    NHITS / NBEATS) can be slotted in without changing callers.
    """
    res = _predict_batch_cpu(models, horizon)
    res.backend = "gpu_fallback_cpu"
    return res


__all__ = ["predict_batch", "BatchInferenceResult"]
