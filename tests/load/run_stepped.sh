#!/usr/bin/env bash
# Runs sequential Locust load tests at escalating user counts.
# Each step saves its own CSV to reports/ and logs to MLflow.
#
# Usage:
#   ./tests/load/run_stepped.sh                    # default steps: 10 20 30 50
#   ./tests/load/run_stepped.sh --users "5 10 20"  # custom steps
#   ./tests/load/run_stepped.sh --time 60s          # shorter runs

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

HOST="${HOST:-http://52.15.187.251:8000}"
RUN_TIME="${RUN_TIME:-120s}"
USER_STEPS="${USER_STEPS:-10 20 30 50}"

# Parse optional flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --users) USER_STEPS="$2"; shift 2;;
    --time)  RUN_TIME="$2";   shift 2;;
    --host)  HOST="$2";       shift 2;;
    *) echo "Unknown flag: $1"; exit 1;;
  esac
done

# Find the next run number
next_run() {
  local max=0
  for f in reports/run*_stats.csv; do
    [[ -e "$f" ]] || continue
    n=$(basename "$f" | grep -o 'run[0-9]*' | grep -o '[0-9]*')
    [[ "$n" -gt "$max" ]] && max="$n"
  done
  echo $((max + 1))
}

PYTHON=".venv/bin/python"
LOCUST=".venv/bin/locust"

echo "============================================================"
echo "  Stepped Load Test"
echo "  Host:  $HOST"
echo "  Steps: $USER_STEPS"
echo "  Time:  $RUN_TIME per step"
echo "============================================================"

for USERS in $USER_STEPS; do
  RUN_NUM=$(next_run)
  CSV_PREFIX="reports/run${RUN_NUM}"
  SPAWN_RATE=$(( USERS / 5 < 1 ? 1 : USERS / 5 ))

  echo ""
  echo "--- Run $RUN_NUM: $USERS users, spawn-rate $SPAWN_RATE, $RUN_TIME ---"

  MLFLOW_TRACKING_URI="$(pwd)/mlflow.db" \
  PYTHONPATH="$(pwd)" \
  "$LOCUST" \
    -f tests/load/locustfile.py \
    --host "$HOST" \
    --users "$USERS" \
    --spawn-rate "$SPAWN_RATE" \
    --run-time "$RUN_TIME" \
    --headless \
    --csv "$CSV_PREFIX" \
    2>&1 || true

  echo "  Saved: ${CSV_PREFIX}_stats.csv"

  # Brief cooldown between steps so the server recovers
  if [[ "$USERS" != "$(echo "$USER_STEPS" | awk '{print $NF}')" ]]; then
    echo "  Cooling down 10s..."
    sleep 10
  fi
done

echo ""
echo "All steps complete. Run the comparison report:"
echo "  python tests/load/compare_runs.py"
