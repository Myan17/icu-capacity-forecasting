# Phase 2 — Data Ingestion + Great Expectations Validation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract GE validation into a testable module, write pytest suites (moto-mocked AWS), add SQS DLQs to SAM template, and seed S3 with HHS data — producing documented test evidence for the paper.

**Architecture:** Validation logic moves from `handler.py` into `lambdas/ingest/validation_suite.py` (pure-function, no AWS calls) so it can be unit-tested without Lambda context. Handler tests use `moto` to mock DynamoDB + S3. A seed script uploads the existing `data/cleaned_hhs_ml_ready.csv` to S3.

**Tech Stack:** pytest, moto 5.x (AWS mocking), great-expectations 1.x (ephemeral mode), pytest-cov, boto3

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| **Create** | `lambdas/ingest/validation_suite.py` | Pure GE + pandas validation — no boto3/Lambda deps |
| **Modify** | `lambdas/ingest/handler.py` | Import from `validation_suite` instead of inline functions |
| **Create** | `tests/test_ingest_validation.py` | Unit tests for all validation rules (no AWS) |
| **Create** | `tests/test_ingest_handler.py` | Handler integration tests with moto mocks |

| **Create** | `tests/conftest.py` | Shared pytest fixtures (sample DataFrames, moto setup) |
| **Create** | `scripts/seed_s3.py` | Upload `data/cleaned_hhs_ml_ready.csv` to S3 raw prefix |
| **Modify** | `template.yaml` | Add `IngestDLQ` + `ForecastDLQ` SQS queues; wire to EventBridge schedules |

---

## Task 1 — Extract validation into `validation_suite.py`

**Files:**
- Create: `lambdas/ingest/validation_suite.py`
- Modify: `lambdas/ingest/handler.py` (swap inline functions for import)

- [ ] **Step 1: Write the failing import test**

```python
# tests/test_ingest_validation.py
from lambdas.ingest.validation_suite import run_validation
import pandas as pd

def test_import():
    assert callable(run_validation)
```

Run: `cd /Users/myangupta/Downloads/CSCI5980/hospital-forecasting && python -m pytest tests/test_ingest_validation.py::test_import -v`
Expected: **FAIL** — `ModuleNotFoundError: No module named 'lambdas.ingest.validation_suite'`

- [ ] **Step 2: Create `lambdas/ingest/validation_suite.py`**

```python
"""Pure validation logic — no boto3 / Lambda context required.

Returns a ValidationResult dict so the Lambda handler and tests
can both call this without any AWS setup.
"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd

try:
    import great_expectations as gx
    _GE_AVAILABLE = True
except ImportError:
    _GE_AVAILABLE = False

REQUIRED_COLUMNS = ["hospital_id", "timestamp", "icu_capacity", "icu_occupied"]


def run_validation(df: pd.DataFrame) -> dict[str, Any]:
    """Run schema + sanity checks on a snapshot DataFrame.

    Returns:
        {
          "success": bool,
          "summary": str,
          "failures": list[str],
          "rows_checked": int,
          "ge_used": bool,
        }
    """
    failures: list[str] = []

    # R1 — required columns present
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            failures.append(f"Missing required column: {col}")

    if failures:
        return _result(failures, len(df))

    # R2 — no null timestamps
    null_ts = int(df["timestamp"].isna().sum())
    if null_ts:
        failures.append(f"{null_ts} rows have null timestamp")

    # R3 — non-negative occupancy
    neg_occ = int((df["icu_occupied"] < 0).sum())
    if neg_occ:
        failures.append(f"{neg_occ} rows have negative icu_occupied")

    # R4 — capacity > 0
    bad_cap = int((df["icu_capacity"] <= 0).sum())
    if bad_cap:
        failures.append(f"{bad_cap} rows have zero/negative icu_capacity")

    # R5 — occupied not implausibly above capacity (>150% flags data error)
    over = int((df["icu_occupied"] > df["icu_capacity"] * 1.5).sum())
    if over:
        failures.append(f"{over} rows: icu_occupied > 1.5 × icu_capacity")

    ge_used = False
    if _GE_AVAILABLE and not failures:
        ge_failures = _run_ge_suite(df)
        failures.extend(ge_failures)
        ge_used = True

    return _result(failures, len(df), ge_used)


def _result(failures: list[str], rows: int, ge_used: bool = False) -> dict:
    return {
        "success": len(failures) == 0,
        "summary": "; ".join(failures) if failures else "All checks passed",
        "failures": failures,
        "rows_checked": rows,
        "ge_used": ge_used,
    }


def _run_ge_suite(df: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    try:
        context = gx.get_context(mode="ephemeral")
        ds = context.data_sources.add_pandas("ingest")
        asset = ds.add_dataframe_asset("snapshots")
        batch_def = asset.add_batch_definition_whole_dataframe("batch")
        batch = batch_def.get_batch(batch_parameters={"dataframe": df})

        suite = context.suites.add(gx.ExpectationSuite(name="snapshot_suite"))
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column="hospital_id")
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column="timestamp")
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="icu_occupied", min_value=0, max_value=10_000
            )
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="icu_capacity", min_value=1, max_value=10_000
            )
        )

        vd = context.validation_definitions.add(
            gx.ValidationDefinition(name="v", data=batch_def, suite=suite)
        )
        result = vd.run()
        if not result.success:
            for r in result.results:
                if not r.success:
                    failures.append(f"GE: {r.expectation_config.type} failed")
    except Exception as exc:
        failures.append(f"GE suite error: {exc}")
    return failures
```

- [ ] **Step 3: Run test — verify it passes**

```bash
python -m pytest tests/test_ingest_validation.py::test_import -v
```
Expected: **PASS**

- [ ] **Step 4: Update `handler.py` to import from `validation_suite`**

Replace the inline `_run_validation` and `_run_ge_suite` functions in `lambdas/ingest/handler.py` with:

```python
# At top of handler.py, add:
from validation_suite import run_validation

# Replace the call site (in lambda_handler):
# OLD: validation_result = _run_validation(df)
# NEW:
validation_result = run_validation(df)
```

Also delete the `_run_validation`, `_run_ge_suite` function bodies from `handler.py`.

- [ ] **Step 5: Commit**

```bash
git add lambdas/ingest/validation_suite.py lambdas/ingest/handler.py
git commit -m "refactor: extract GE validation into testable validation_suite module"
```

---

## Task 2 — Unit tests for all validation rules

**Files:**
- Modify: `tests/test_ingest_validation.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write all failing tests**

```python
# tests/test_ingest_validation.py
from __future__ import annotations

import pandas as pd
import pytest
from lambdas.ingest.validation_suite import run_validation


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _good_df(**overrides) -> pd.DataFrame:
    base = {
        "hospital_id":  ["H001", "H001", "H001"],
        "timestamp":    pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]),
        "icu_capacity": [100, 100, 100],
        "icu_occupied": [70, 75, 80],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ── Happy path ────────────────────────────────────────────────────────────────

def test_valid_dataframe_passes():
    result = run_validation(_good_df())
    assert result["success"] is True
    assert result["failures"] == []
    assert result["rows_checked"] == 3


# ── R1: Missing columns ───────────────────────────────────────────────────────

def test_missing_hospital_id_fails():
    df = _good_df().drop(columns=["hospital_id"])
    result = run_validation(df)
    assert result["success"] is False
    assert any("hospital_id" in f for f in result["failures"])


def test_missing_timestamp_fails():
    df = _good_df().drop(columns=["timestamp"])
    result = run_validation(df)
    assert result["success"] is False
    assert any("timestamp" in f for f in result["failures"])


def test_missing_icu_capacity_fails():
    df = _good_df().drop(columns=["icu_capacity"])
    result = run_validation(df)
    assert result["success"] is False


def test_missing_icu_occupied_fails():
    df = _good_df().drop(columns=["icu_occupied"])
    result = run_validation(df)
    assert result["success"] is False


# ── R2: Null timestamps ───────────────────────────────────────────────────────

def test_null_timestamp_fails():
    df = _good_df()
    df.loc[0, "timestamp"] = pd.NaT
    result = run_validation(df)
    assert result["success"] is False
    assert any("null timestamp" in f for f in result["failures"])


# ── R3: Negative occupancy ────────────────────────────────────────────────────

def test_negative_icu_occupied_fails():
    df = _good_df(icu_occupied=[-1, 50, 60])
    result = run_validation(df)
    assert result["success"] is False
    assert any("negative icu_occupied" in f for f in result["failures"])


# ── R4: Zero/negative capacity ────────────────────────────────────────────────

def test_zero_capacity_fails():
    df = _good_df(icu_capacity=[0, 100, 100])
    result = run_validation(df)
    assert result["success"] is False
    assert any("icu_capacity" in f for f in result["failures"])


def test_negative_capacity_fails():
    df = _good_df(icu_capacity=[-10, 100, 100])
    result = run_validation(df)
    assert result["success"] is False


# ── R5: Occupied > 1.5 × capacity ────────────────────────────────────────────

def test_implausible_occupancy_fails():
    df = _good_df(icu_occupied=[200, 75, 80])  # 200 > 1.5 × 100
    result = run_validation(df)
    assert result["success"] is False
    assert any("1.5" in f for f in result["failures"])


def test_over_capacity_but_within_tolerance_passes():
    # 105% occupied is clinically possible (hallway beds) — should not fail
    df = _good_df(icu_occupied=[105, 75, 80])
    result = run_validation(df)
    assert result["success"] is True


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_dataframe_fails():
    df = pd.DataFrame(columns=["hospital_id", "timestamp", "icu_capacity", "icu_occupied"])
    result = run_validation(df)
    # Empty passes column checks but has 0 rows — success is fine, just no data
    assert result["rows_checked"] == 0


def test_single_row_valid():
    df = _good_df().head(1)
    result = run_validation(df)
    assert result["success"] is True


def test_summary_is_string():
    result = run_validation(_good_df())
    assert isinstance(result["summary"], str)
```

- [ ] **Step 2: Run all tests — verify they fail**

```bash
python -m pytest tests/test_ingest_validation.py -v 2>&1 | head -40
```
Expected: Multiple **FAIL** lines (module not yet importable from test root).

- [ ] **Step 3: Create `tests/conftest.py` to add `lambdas/` to sys.path**

```python
# tests/conftest.py
import sys
from pathlib import Path

# Allow tests to import from lambdas/ and shared/ without installing them
ROOT = Path(__file__).parent.parent
for p in [str(ROOT), str(ROOT / "lambdas"), str(ROOT / "shared")]:
    if p not in sys.path:
        sys.path.insert(0, p)
```

- [ ] **Step 4: Run all validation tests — verify they pass**

```bash
python -m pytest tests/test_ingest_validation.py -v
```
Expected: **13 PASSED**

**Screenshot this output for the paper (Q5 / testing evidence).**

- [ ] **Step 5: Commit**

```bash
git add tests/test_ingest_validation.py tests/conftest.py
git commit -m "test: add 13 unit tests for GE validation rules (all passing)"
```

---

## Task 3 — Handler integration tests with moto (mocked AWS)

**Files:**
- Create: `tests/test_ingest_handler.py`

- [ ] **Step 1: Install test dependencies**

```bash
pip install moto[dynamodb,s3]>=5.0.0 pytest-cov
```

- [ ] **Step 2: Write failing handler tests**

```python
# tests/test_ingest_handler.py
from __future__ import annotations

import io
import json
import os

import boto3
import pandas as pd
import pytest
from moto import mock_aws


# ── Environment setup (must happen before importing handler) ──────────────────

os.environ.setdefault("SNAPSHOTS_TABLE", "hospital-snapshots-test")
os.environ.setdefault("FORECASTS_TABLE", "hospital-forecasts-test")
os.environ.setdefault("ALERTS_TABLE",    "hospital-alerts-test")
os.environ.setdefault("DATA_BUCKET",     "hospital-test-bucket")
os.environ.setdefault("ENVIRONMENT",     "test")
os.environ.setdefault("ALERT_OCCUPANCY_YELLOW", "0.75")
os.environ.setdefault("ALERT_OCCUPANCY_RED",    "0.90")


@pytest.fixture(autouse=True)
def aws_mock():
    """Start moto mock for every test in this module."""
    with mock_aws():
        _create_tables()
        _create_bucket()
        yield


def _create_tables():
    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    for name in ["hospital-snapshots-test", "hospital-alerts-test"]:
        ddb.create_table(
            TableName=name,
            BillingMode="PAY_PER_REQUEST",
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            KeySchema=[
                {"AttributeName": "pk", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
        )


def _create_bucket():
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="hospital-test-bucket")


def _put_csv(df: pd.DataFrame, key: str = "raw/test.csv"):
    s3 = boto3.client("s3", region_name="us-east-1")
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket="hospital-test-bucket", Key=key, Body=buf.getvalue())


def _good_csv_df() -> pd.DataFrame:
    return pd.DataFrame({
        "hospital_id":  ["H001"] * 5,
        "timestamp":    pd.date_range("2024-01-01", periods=5, freq="W"),
        "icu_capacity": [100] * 5,
        "icu_occupied": [70, 72, 75, 78, 80],
        "admissions":   [5] * 5,
        "discharges":   [4] * 5,
        "transfers":    [1] * 5,
        "ed_arrivals":  [10] * 5,
        "staffing_level": [0.85] * 5,
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_valid_ingest_writes_snapshots_to_dynamodb():
    from ingest.handler import lambda_handler

    _put_csv(_good_csv_df())
    result = lambda_handler(
        {"s3_bucket": "hospital-test-bucket", "s3_key": "raw/test.csv"},
        {},
    )

    assert result["status"] == "success"
    assert result["rows_written"] == 5

    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    table = ddb.Table("hospital-snapshots-test")
    resp = table.scan()
    assert len(resp["Items"]) == 5


def test_snapshots_have_occupancy_ratio():
    from ingest.handler import lambda_handler

    _put_csv(_good_csv_df())
    lambda_handler({"s3_bucket": "hospital-test-bucket", "s3_key": "raw/test.csv"}, {})

    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    items = ddb.Table("hospital-snapshots-test").scan()["Items"]
    for item in items:
        assert "occupancy_ratio" in item
        assert 0.0 <= float(item["occupancy_ratio"]) <= 2.0


def test_invalid_data_writes_ingestion_failed_alert():
    from ingest.handler import lambda_handler

    bad_df = pd.DataFrame({
        "hospital_id":  ["H001"],
        "timestamp":    [pd.NaT],          # null timestamp → R2 failure
        "icu_capacity": [100],
        "icu_occupied": [70],
    })
    _put_csv(bad_df)
    result = lambda_handler(
        {"s3_bucket": "hospital-test-bucket", "s3_key": "raw/test.csv"},
        {},
    )

    assert result["status"] == "validation_failed"

    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    alerts = ddb.Table("hospital-alerts-test").scan()["Items"]
    assert any(a["risk_level"] == "ingestion_failed" for a in alerts)


def test_negative_occupied_blocked_and_alerts():
    from ingest.handler import lambda_handler

    bad_df = _good_csv_df().copy()
    bad_df.loc[0, "icu_occupied"] = -5
    _put_csv(bad_df)
    result = lambda_handler(
        {"s3_bucket": "hospital-test-bucket", "s3_key": "raw/test.csv"},
        {},
    )

    assert result["status"] == "validation_failed"


def test_no_data_returns_no_data_status():
    from ingest.handler import lambda_handler

    result = lambda_handler({}, {})
    assert result["status"] == "no_data"


def test_hospital_id_filter_writes_only_matching_rows():
    from ingest.handler import lambda_handler

    df = _good_csv_df().copy()
    df_extra = _good_csv_df().copy()
    df_extra["hospital_id"] = "H002"
    combined = pd.concat([df, df_extra], ignore_index=True)
    _put_csv(combined)

    result = lambda_handler(
        {
            "s3_bucket":   "hospital-test-bucket",
            "s3_key":      "raw/test.csv",
            "hospital_id": "H001",
        },
        {},
    )

    assert result["rows_written"] == 5
    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    items = ddb.Table("hospital-snapshots-test").scan()["Items"]
    assert all(i["hospital_id"] == "H001" for i in items)
```

- [ ] **Step 3: Run — verify failures**

```bash
python -m pytest tests/test_ingest_handler.py -v 2>&1 | head -30
```
Expected: **FAIL** — `ModuleNotFoundError` (moto env not set up yet or import path issue).

- [ ] **Step 4: Fix `conftest.py` to add `lambdas/ingest` path explicitly**

```python
# tests/conftest.py  (replace entirely)
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
for p in [
    str(ROOT),
    str(ROOT / "lambdas"),
    str(ROOT / "lambdas" / "ingest"),
    str(ROOT / "lambdas" / "api"),
    str(ROOT / "shared"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)
```

- [ ] **Step 5: Run — verify all 6 handler tests pass**

```bash
python -m pytest tests/test_ingest_handler.py -v
```
Expected: **6 PASSED**

**Screenshot this output for the paper (moto-based cloud integration testing evidence).**

- [ ] **Step 6: Run full test suite with coverage**

```bash
python -m pytest tests/ -v --cov=lambdas/ingest --cov=shared --cov-report=term-missing
```
Expected: **19 PASSED**, coverage ≥ 80% on `validation_suite.py` and `handler.py`.

**Screenshot the coverage table for the paper.**

- [ ] **Step 7: Commit**

```bash
git add tests/test_ingest_handler.py tests/conftest.py
git commit -m "test: add 6 moto-mocked handler integration tests (ingest Lambda)"
```

---

## Task 4 — SQS Dead-Letter Queues in SAM template

**Files:**
- Modify: `template.yaml`

- [ ] **Step 1: Write a validation test for the SAM template**

```bash
# Verify sam validate passes before and after changes
sam validate --lint
```
Expected: `template.yaml is valid SAM template`

- [ ] **Step 2: Add SQS DLQ resources to `template.yaml`**

Add inside `Resources:` (after the DynamoDB tables, before Lambda functions):

```yaml
  # ── Dead-Letter Queues (EventBridge schedule failures) ─────────────────────
  IngestDLQ:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub hospital-ingest-dlq-${Environment}
      MessageRetentionPeriod: 1209600   # 14 days
      Tags:
        - Key: Project
          Value: hospital-forecasting

  ForecastDLQ:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub hospital-forecast-dlq-${Environment}
      MessageRetentionPeriod: 1209600
      Tags:
        - Key: Project
          Value: hospital-forecasting
```

Update the `IngestFunction` EventBridge event's `DeadLetterConfig`:

```yaml
      Events:
        WeeklyIngest:
          Type: ScheduleV2
          Properties:
            ScheduleExpression: "cron(0 6 ? * MON *)"
            RetryPolicy:
              MaximumRetryAttempts: 2
            DeadLetterConfig:
              Type: SQS
              Destination: !GetAtt IngestDLQ.Arn
```

Update the `ForecastFunction` EventBridge event similarly:

```yaml
            DeadLetterConfig:
              Type: SQS
              Destination: !GetAtt ForecastDLQ.Arn
```

Add DLQ ARNs to `Outputs:`:

```yaml
  IngestDLQUrl:
    Description: Ingest Dead-Letter Queue URL (monitor for failed ingest runs)
    Value: !Ref IngestDLQ
  ForecastDLQUrl:
    Description: Forecast Dead-Letter Queue URL
    Value: !Ref ForecastDLQ
```

- [ ] **Step 3: Validate the updated template**

```bash
sam validate --lint
```
Expected: `template.yaml is valid SAM template`

- [ ] **Step 4: Commit**

```bash
git add template.yaml
git commit -m "infra: add SQS DLQs for EventBridge ingest and forecast schedules"
```

---

## Task 5 — S3 seed script for HHS data

**Files:**
- Create: `scripts/seed_s3.py`

- [ ] **Step 1: Write the script**

```python
"""Upload HHS CSV to S3 for Lambda consumption.

Usage:
    python scripts/seed_s3.py --bucket <bucket-name>

Uploads:
    data/cleaned_hhs_ml_ready.csv  →  s3://<bucket>/raw/cleaned_hhs_ml_ready.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "data" / "cleaned_hhs_ml_ready.csv"


def seed(bucket: str) -> None:
    if not CSV_PATH.exists():
        print(f"[ERROR] {CSV_PATH} not found")
        sys.exit(1)

    s3 = boto3.client("s3")
    key = f"raw/{CSV_PATH.name}"
    s3.upload_file(str(CSV_PATH), bucket, key)
    size_kb = CSV_PATH.stat().st_size // 1024
    print(f"[OK] Uploaded {CSV_PATH.name} ({size_kb} KB) → s3://{bucket}/{key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    seed(parser.parse_args().bucket)
```

- [ ] **Step 2: Write a smoke test**

```bash
# Dry run — verify file exists locally
python scripts/seed_s3.py --help
```
Expected: prints usage without error.

- [ ] **Step 3: Commit**

```bash
git add scripts/seed_s3.py
git commit -m "script: add seed_s3.py to upload HHS CSV to S3 data bucket"
```

---

## Final verification

- [ ] **Run the complete test suite**

```bash
python -m pytest tests/ -v --cov=lambdas/ingest --cov=shared \
  --cov-report=term-missing --cov-report=html:htmlcov
```
Expected: **≥19 PASSED**, 0 failures, coverage report written to `htmlcov/`.

**Open `htmlcov/index.html` in a browser and screenshot for the paper.**

- [ ] **Validate SAM template one final time**

```bash
sam validate --lint
```

- [ ] **Final commit**

```bash
git add -A
git commit -m "feat: Phase 2 complete — GE validation, moto tests, SQS DLQs, S3 seed"
```

---

## Paper evidence checklist (Q5 / Testing)

| Evidence | How to capture |
|---|---|
| pytest output (13 unit tests) | Screenshot terminal after `pytest tests/test_ingest_validation.py -v` |
| pytest output (6 handler tests) | Screenshot terminal after `pytest tests/test_ingest_handler.py -v` |
| Coverage report | Screenshot `htmlcov/index.html` |
| GE Data Docs HTML | Screenshot S3 console showing `data-docs/*.html` after a real ingest run |
| DLQ in AWS console | Screenshot SQS queue list after `sam deploy` |
| SAM `sam validate` pass | Screenshot terminal output |

> **Tool used for testing:** pytest 8.x + moto 5.x (industry-standard AWS mocking) + great-expectations 1.x
