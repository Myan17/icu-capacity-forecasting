"""Pure validation logic — no AWS dependencies.

Extracted from handler.py so it can be unit-tested without Lambda context.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

try:
    import great_expectations as gx
    _GE_AVAILABLE = True
except ImportError:
    _GE_AVAILABLE = False

REQUIRED_COLUMNS = [
    "hospital_id", "timestamp", "icu_capacity", "icu_occupied",
]


def run_validation(df: pd.DataFrame, *, use_ge: bool | None = None) -> dict:
    """Run schema + sanity checks on a snapshot DataFrame.

    Args:
        df: Normalized snapshot DataFrame.
        use_ge: Override GE availability flag (used in tests).

    Returns:
        dict with keys: success, summary, failures, rows_checked.
    """
    failures: list[str] = []

    # 1. Required columns present
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            failures.append(f"Missing required column: {col}")

    if failures:
        return {"success": False, "summary": "; ".join(failures), "failures": failures}

    # 2. No missing timestamps
    null_ts = df["timestamp"].isna().sum()
    if null_ts > 0:
        failures.append(f"{null_ts} rows have null timestamp")

    # 3. Non-negative occupancy
    neg_occ = (df["icu_occupied"] < 0).sum()
    if neg_occ > 0:
        failures.append(f"{neg_occ} rows have negative icu_occupied")

    # 4. Implausible occupancy ratio (> 1.5× capacity)
    if "icu_capacity" in df.columns:
        over = (df["icu_occupied"] > df["icu_capacity"] * 1.5).sum()
        if over > 0:
            failures.append(
                f"{over} rows have icu_occupied > 1.5 × icu_capacity (data quality issue)"
            )

    # 5. Non-positive capacity
    neg_cap = (df["icu_capacity"] <= 0).sum()
    if neg_cap > 0:
        failures.append(f"{neg_cap} rows have zero or negative icu_capacity")

    _use_ge = _GE_AVAILABLE if use_ge is None else use_ge
    if _use_ge:
        ge_failures = run_ge_suite(df)
        failures.extend(ge_failures)

    success = len(failures) == 0
    return {
        "success": success,
        "summary": "; ".join(failures) if failures else "All checks passed",
        "failures": failures,
        "rows_checked": len(df),
    }


def run_ge_suite(df: pd.DataFrame) -> list[str]:
    """Run Great Expectations expectations and return failure messages."""
    failures: list[str] = []
    try:
        context = gx.get_context(mode="ephemeral")
        data_source = context.data_sources.add_pandas("ingest_pandas")
        data_asset = data_source.add_dataframe_asset("snapshot_df")
        batch_def = data_asset.add_batch_definition_whole_dataframe("batch")

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

        validation_def = context.validation_definitions.add(
            gx.ValidationDefinition(
                name="snapshot_validation",
                data=batch_def,
                suite=suite,
            )
        )
        result = validation_def.run()
        if not result.success:
            for res in result.results:
                if not res.success:
                    failures.append(f"GE: {res.expectation_config.type} failed")
    except Exception as exc:
        logger.warning("GE suite error (non-fatal): %s", exc)
    return failures
