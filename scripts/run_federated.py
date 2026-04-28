"""Run a federated-learning simulation over the synthetic hospital fleet.

Implements all four L4 features in one script:
  * FL simulation (per-hospital local training, no raw data sharing)
  * Per-hospital tuning (PerHospitalTuner over a tier-aware grid)
  * TiFL tiering (HospitalProfile.tifl_tier)
  * FedAT (FedATCoordinator with staleness-decayed weights)
  * Weighted aggregation (weighted_fedavg via FedAT)
  * Async update cadence (period 1 / 2 / 4 by tier)

Optionally logs per-round metrics to MLflow when ``--mlflow`` is passed.

Usage:
    python -m scripts.run_federated --hospitals 8 --rounds 6 --strategy weighted
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ml.federated.client import FederatedClient
from ml.federated.fedat import FedATCoordinator, tier_summary
from ml.per_hospital.tuner import PerHospitalTuner
from shared.hospital_tiers import HospitalProfile, profile_hospital


def _synthetic_hospitals(n: int, seed: int = 0) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out: Dict[str, pd.DataFrame] = {}
    for i in range(n):
        # Spread row counts across small / medium / large tiers
        rows = int(rng.integers(15, 220))
        capacity = int(rng.integers(40, 200))
        baseline = float(rng.uniform(0.4, 0.85)) * capacity
        noise = float(rng.uniform(2, 12))
        timestamps = pd.date_range("2023-01-01", periods=rows, freq="W")
        occ = baseline + 6 * np.sin(np.arange(rows) * 0.25) + rng.normal(0, noise, rows)
        out[f"hosp_{i:03d}"] = pd.DataFrame({
            "timestamp":    timestamps,
            "icu_occupied": np.clip(occ, 1, capacity),
            "icu_capacity": capacity,
        })
    return out


def _build_clients(hospitals: Dict[str, pd.DataFrame],
                   profiles: Dict[str, HospitalProfile]) -> List[FederatedClient]:
    clients: List[FederatedClient] = []
    for hid, df in hospitals.items():
        prof = profiles[hid]
        # Tier-aware local epochs: bigger tiers train longer per round
        if prof.workload_tier.value == "large":
            local_epochs = 5
        elif prof.workload_tier.value == "medium":
            local_epochs = 3
        else:
            local_epochs = 2
        clients.append(FederatedClient(prof, local_epochs=local_epochs).load_data(df))
    return clients


def _per_hospital_tuning_summary(hospitals: Dict[str, pd.DataFrame],
                                 profiles: Dict[str, HospitalProfile]) -> Dict[str, Any]:
    """Run a small per-hospital tuner so the FL run starts from sensible
    classical-model RMSEs, included in the report so the FL improvement is
    contextualised against per-hospital tuning (Lecture 4 non-IID baseline)."""
    out: Dict[str, Any] = {}
    tuner = PerHospitalTuner(horizon=4)
    for hid, df in hospitals.items():
        try:
            res = tuner.tune(profiles[hid], df)
            out[hid] = res.to_dict()
        except Exception as exc:
            out[hid] = {"error": str(exc)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hospitals", type=int, default=8)
    ap.add_argument("--rounds",    type=int, default=6)
    ap.add_argument("--strategy",
                    choices=["fedavg", "weighted", "fedavgm"],
                    default="weighted")
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--horizon",   type=int, default=4)
    ap.add_argument("--mlflow",    action="store_true")
    ap.add_argument("--skip-tuning", action="store_true")
    ap.add_argument("--json",      action="store_true")
    args = ap.parse_args()

    hospitals = _synthetic_hospitals(args.hospitals, seed=args.seed)
    profiles = {hid: profile_hospital(hid, df) for hid, df in hospitals.items()}
    clients = _build_clients(hospitals, profiles)

    coord = FedATCoordinator(clients=clients, staleness_alpha=0.5)
    state, round_logs = coord.run(num_rounds=args.rounds, strategy=args.strategy)

    tuning_summary = (
        _per_hospital_tuning_summary(hospitals, profiles)
        if not args.skip_tuning else {}
    )

    fl_evaluations = coord.evaluate(state, horizon=args.horizon)
    tiers = tier_summary(round_logs, profiles)

    report: Dict[str, Any] = {
        "config": {
            "hospitals":     args.hospitals,
            "rounds":        args.rounds,
            "strategy":      args.strategy,
            "horizon":       args.horizon,
            "seed":          args.seed,
        },
        "tier_distribution": {
            tier: sum(1 for p in profiles.values() if p.tifl_tier.value == tier)
            for tier in ("tier_1", "tier_2", "tier_3")
        },
        "rounds":           [r.to_dict() for r in round_logs],
        "tier_summary":     tiers,
        "global_theta":     state.theta.tolist(),
        "fl_predictions":   fl_evaluations,
        "per_hospital_tuning": tuning_summary,
    }

    if args.mlflow:
        try:
            import mlflow

            mlflow.set_experiment("federated-learning-sim")
            with mlflow.start_run():
                mlflow.log_params(report["config"])
                for r in round_logs:
                    mlflow.log_metric("avg_loss", r.avg_loss, step=r.round_num)
                    mlflow.log_metric("participants", len(r.participants), step=r.round_num)
                for tier, stats in tiers.items():
                    if not np.isnan(stats["mean_loss"]):
                        mlflow.log_metric(f"{tier}_mean_loss", stats["mean_loss"])
                mlflow.log_dict(report, "federated_run.json")
        except Exception as exc:
            print(f"[warn] MLflow logging failed: {exc}")

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return

    print(f"=== Federated learning simulation ({args.strategy}) ===\n")
    print("Tier distribution:", report["tier_distribution"], "\n")
    print("Per-round summary:")
    for r in round_logs:
        print(f"  R{r.round_num:>2}  parts={len(r.participants):>2}  "
              f"avg_loss={r.avg_loss:>7.3f}  duration={r.duration_ms:>6.2f}ms")
    print("\nTier summary:")
    for tier, stats in tiers.items():
        loss_str = (f"{stats['mean_loss']:.3f}" if not np.isnan(stats["mean_loss"]) else "—")
        print(f"  {tier:<7}  rounds_participated={stats['rounds_participated']:>3}  mean_loss={loss_str}")
    print(f"\nFinal global θ = {state.theta.round(4).tolist()}")
    if tuning_summary:
        print("\nBest per-hospital classical RMSE:")
        for hid, info in tuning_summary.items():
            if "error" in info:
                print(f"  {hid}: error -> {info['error']}")
            else:
                print(f"  {hid}: {info['model_name']:<8}  rmse={info['rmse']:.3f}  params={info['params']}")


if __name__ == "__main__":
    main()
