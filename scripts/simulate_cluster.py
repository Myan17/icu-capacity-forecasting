"""Local multi-node execution cluster simulator (Lecture 2: Borg/K8s).

Runs the priority scheduler and execution-pool packer end-to-end with synthetic
hospital data, simulating a Borg-style cluster of N worker nodes that each
process a shard of hospitals in parallel.

Usage:
    python -m scripts.simulate_cluster --hospitals 20 --nodes 4 --rounds 3
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from shared.hospital_tiers import HospitalProfile, profile_hospital
from shared.priority_queue import (
    ExecutionPool,
    HospitalSignal,
    PriorityScheduler,
    Shard,
    estimated_makespan_seconds,
    utilisation_balance,
)
from shared.workload_metrics import WorkloadKind, WorkloadProfiler


def _synthetic_hospitals(n: int, seed: int = 0) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out: Dict[str, pd.DataFrame] = {}
    for i in range(n):
        rows = int(rng.integers(15, 220))
        capacity = int(rng.integers(40, 200))
        baseline = float(rng.uniform(0.4, 0.85)) * capacity
        noise = float(rng.uniform(2, 12))
        timestamps = pd.date_range("2023-01-01", periods=rows, freq="W")
        occ = baseline + 5 * np.sin(np.arange(rows) * 0.3) + rng.normal(0, noise, rows)
        out[f"hosp_{i:03d}"] = pd.DataFrame({
            "timestamp":    timestamps,
            "icu_occupied": np.clip(occ, 1, capacity),
            "icu_capacity": capacity,
        })
    return out


def _signal_for(hid: str, df: pd.DataFrame, profile: HospitalProfile) -> HospitalSignal:
    last = float(df["icu_occupied"].iloc[-1])
    cap = float(df["icu_capacity"].iloc[-1] or 1)
    occ_ratio = last / max(cap, 1.0)
    last_breach = max(0.0, min(1.0, (occ_ratio - 0.6) * 1.5))  # cheap heuristic
    return HospitalSignal(
        profile=profile,
        occupancy_ratio=occ_ratio,
        last_breach_prob=last_breach,
    )


def _process_shard(shard: Shard, hospitals: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Pretend each hospital takes ~10ms × workload_tier weight to forecast."""
    profiler = WorkloadProfiler()
    finished = []
    with profiler.stage(f"shard_{shard.index}", WorkloadKind.CPU):
        for item in shard.items:
            df = hospitals.get(item.hospital_id)
            if df is None:
                continue
            # Mimic forecast time by sleeping proportional to row count
            time.sleep(min(0.001 + len(df) * 1e-5, 0.05))
            finished.append(item.hospital_id)
    return {
        "shard": shard.index,
        "hospitals": finished,
        "duration_ms": profiler.report().total_ms(),
        "shard_score": shard.total_score,
    }


def run_simulation(num_hospitals: int, num_nodes: int, rounds: int, seed: int = 0) -> Dict[str, Any]:
    hospitals = _synthetic_hospitals(num_hospitals, seed=seed)
    profiles = {hid: profile_hospital(hid, df) for hid, df in hospitals.items()}

    round_logs: List[Dict[str, Any]] = []
    for r in range(1, rounds + 1):
        rng = random.Random(seed + r)
        # Simulate occupancy drift each round
        for hid, df in hospitals.items():
            jitter = rng.uniform(-0.05, 0.05)
            df["icu_occupied"] = (df["icu_occupied"].astype(float) * (1.0 + jitter)).clip(lower=1.0)

        sched = PriorityScheduler()
        for hid, df in hospitals.items():
            sched.submit(_signal_for(hid, df, profiles[hid]))

        prioritised = list(sched.drain())
        shards = ExecutionPool(num_shards=num_nodes).pack(prioritised)
        balance = utilisation_balance(shards)
        makespan = estimated_makespan_seconds(shards, per_hospital_seconds=0.05)

        wall_start = time.perf_counter()
        per_shard: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=num_nodes) as pool:
            futures = [pool.submit(_process_shard, s, hospitals) for s in shards]
            for fut in as_completed(futures):
                per_shard.append(fut.result())
        wall_ms = (time.perf_counter() - wall_start) * 1000.0

        round_logs.append({
            "round":           r,
            "wall_ms":         round(wall_ms, 2),
            "balance":         round(balance, 3),
            "estimated_makespan_s": round(makespan, 3),
            "top5_priority":   [
                {"hospital": p.hospital_id, "score": round(p.score, 3), "reason": p.reason}
                for p in prioritised[:5]
            ],
            "shards": sorted(per_shard, key=lambda s: s["shard"]),
        })

    return {
        "num_hospitals": num_hospitals,
        "num_nodes":     num_nodes,
        "rounds":        rounds,
        "rounds_log":    round_logs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hospitals", type=int, default=20)
    ap.add_argument("--nodes",     type=int, default=4)
    ap.add_argument("--rounds",    type=int, default=3)
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    args = ap.parse_args()

    result = run_simulation(args.hospitals, args.nodes, args.rounds, seed=args.seed)
    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"=== Cluster simulation: {args.hospitals} hospitals × {args.nodes} nodes × {args.rounds} rounds ===\n")
    for r_log in result["rounds_log"]:
        print(f"Round {r_log['round']}:")
        print(f"  wall_clock        : {r_log['wall_ms']:8.2f} ms")
        print(f"  balance           : {r_log['balance']:.3f} (1.0 = perfect)")
        print(f"  estimated_makespan: {r_log['estimated_makespan_s']:.3f} s")
        print( "  top5 priority     :")
        for p in r_log["top5_priority"]:
            print(f"     {p['hospital']:>10}  score={p['score']:.3f}  reason={p['reason']}")
        print( "  shard distribution:")
        for s in r_log["shards"]:
            print(f"     shard {s['shard']}: {len(s['hospitals'])} hospitals  load={s['shard_score']:.2f}  duration={s['duration_ms']:.1f}ms")
        print()


if __name__ == "__main__":
    main()
