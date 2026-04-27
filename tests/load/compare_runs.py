"""Compare all Locust run CSVs side-by-side.

Reads reports/run*_stats.csv, extracts the Aggregated row, and prints
a summary table plus per-endpoint breakdown for the latest N runs.

Usage:
    python tests/load/compare_runs.py             # all runs
    python tests/load/compare_runs.py --last 4   # last 4 runs only
    python tests/load/compare_runs.py --endpoint /forecast  # filter endpoint
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def load_run(path: Path) -> dict:
    """Return aggregated + per-endpoint rows from a stats CSV."""
    rows = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["Name"]] = row
    return rows


def fmt_ms(v: str) -> str:
    try:
        return f"{float(v):.0f}ms"
    except (ValueError, TypeError):
        return v or "—"


def fmt_pct(v: str) -> str:
    try:
        pct = float(v) * 100
        return f"{pct:.1f}%"
    except (ValueError, TypeError):
        return v or "—"


def fmt_rps(v: str) -> str:
    try:
        return f"{float(v):.2f}"
    except (ValueError, TypeError):
        return v or "—"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last",     type=int, default=0,  help="Show only last N runs")
    parser.add_argument("--endpoint", type=str, default="", help="Filter to a specific endpoint name substring")
    args = parser.parse_args()

    reports = sorted(Path("reports").glob("run*_stats.csv"), key=lambda p: int("".join(filter(str.isdigit, p.stem.split("_stats")[0]))))
    if not reports:
        print("No run CSV files found in reports/")
        sys.exit(1)

    if args.last:
        reports = reports[-args.last:]

    print("\n" + "=" * 80)
    print("  LOAD TEST COMPARISON — AGGREGATED")
    print("=" * 80)

    header = f"{'Run':<8} {'Reqs':>6} {'Fail':>5} {'Fail%':>6} {'Median':>8} {'Avg':>8} {'P95':>8} {'P99':>8} {'RPS':>7}"
    print(header)
    print("-" * 80)

    for p in reports:
        run_num = "".join(filter(str.isdigit, p.stem.split("_stats")[0]))
        data = load_run(p)
        agg = data.get("Aggregated") or data.get("")
        if not agg:
            print(f"run{run_num:<5} (no Aggregated row)")
            continue

        total   = agg["Request Count"]
        fails   = agg["Failure Count"]
        fail_rt = float(fails) / max(float(total), 1)
        median  = agg["50%"]
        avg     = agg["Average Response Time"]
        p95     = agg["95%"]
        p99     = agg["99%"]
        rps     = agg["Requests/s"]

        fail_flag = " !" if float(fails) > 0 else ""
        print(f"run{run_num:<5} {total:>6} {fails:>5} {fmt_pct(str(fail_rt)):>6} "
              f"{fmt_ms(median):>8} {fmt_ms(avg):>8} {fmt_ms(p95):>8} {fmt_ms(p99):>8} {fmt_rps(rps):>7}{fail_flag}")

    print()

    # Per-endpoint breakdown for the latest run
    latest_path = reports[-1]
    latest_run  = "".join(filter(str.isdigit, latest_path.stem.split("_stats")[0]))
    data = load_run(latest_path)

    filter_str = args.endpoint.lower()
    endpoints = {k: v for k, v in data.items() if k not in ("Aggregated", "") and filter_str in k.lower()}

    if not endpoints:
        return

    print(f"  ENDPOINT BREAKDOWN — run{latest_run}")
    print("=" * 80)
    ep_header = f"{'Endpoint':<40} {'Reqs':>5} {'Fail':>5} {'Med':>7} {'P95':>7} {'P99':>7} {'RPS':>6}"
    print(ep_header)
    print("-" * 80)

    for name, row in sorted(endpoints.items()):
        label = f"{row['Type']} {name}"[:39]
        total = row["Request Count"]
        fails = row["Failure Count"]
        fail_flag = " !" if float(fails) > 0 else ""
        print(f"{label:<40} {total:>5} {fails:>5} {fmt_ms(row['50%']):>7} {fmt_ms(row['95%']):>7} {fmt_ms(row['99%']):>7} {fmt_rps(row['Requests/s']):>6}{fail_flag}")

    print()

    # Highlight slow endpoints (P95 > 1000ms)
    slow = {k: v for k, v in endpoints.items() if float(v["95%"]) > 1000}
    if slow:
        print(f"  SLOW ENDPOINTS (P95 > 1s) in run{latest_run}:")
        for name, row in sorted(slow.items(), key=lambda x: -float(x[1]["95%"])):
            print(f"  {row['Type']:4} {name:<38}  P95={fmt_ms(row['95%'])}  P99={fmt_ms(row['99%'])}")
        print()


if __name__ == "__main__":
    main()
