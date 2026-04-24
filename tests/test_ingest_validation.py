"""Unit tests for lambdas/ingest/validation_suite.py.

All tests run without AWS credentials or GE installed (use_ge=False).
"""
from __future__ import annotations

import pandas as pd
import pytest

from lambdas.ingest.validation_suite import run_validation


# ── import sanity ─────────────────────────────────────────────────────────────

def test_import_run_validation():
    assert callable(run_validation)


# ── happy path ────────────────────────────────────────────────────────────────

def test_valid_df_passes(good_df):
    result = run_validation(good_df, use_ge=False)
    assert result["success"] is True
    assert result["failures"] == []
    assert result["rows_checked"] == len(good_df)


def test_summary_message_on_pass(good_df):
    result = run_validation(good_df, use_ge=False)
    assert result["summary"] == "All checks passed"


# ── rule 1: missing required columns ──────────────────────────────────────────

def test_missing_column_fails(missing_col_df):
    result = run_validation(missing_col_df, use_ge=False)
    assert result["success"] is False
    assert any("icu_capacity" in f for f in result["failures"])


def test_missing_hospital_id_fails(good_df):
    df = good_df.drop(columns=["hospital_id"])
    result = run_validation(df, use_ge=False)
    assert result["success"] is False
    assert any("hospital_id" in f for f in result["failures"])


def test_missing_timestamp_col_fails(good_df):
    df = good_df.drop(columns=["timestamp"])
    result = run_validation(df, use_ge=False)
    assert result["success"] is False
    assert any("timestamp" in f for f in result["failures"])


# ── rule 2: null timestamps ───────────────────────────────────────────────────

def test_null_timestamp_value_fails(null_timestamp_df):
    result = run_validation(null_timestamp_df, use_ge=False)
    assert result["success"] is False
    assert any("null timestamp" in f for f in result["failures"])


def test_all_null_timestamps_fail():
    df = pd.DataFrame({
        "hospital_id":  ["H001"],
        "timestamp":    [None],
        "icu_capacity": [100],
        "icu_occupied": [50],
    })
    result = run_validation(df, use_ge=False)
    assert result["success"] is False


# ── rule 3: negative occupancy ────────────────────────────────────────────────

def test_negative_occupied_fails(negative_occupied_df):
    result = run_validation(negative_occupied_df, use_ge=False)
    assert result["success"] is False
    assert any("negative icu_occupied" in f for f in result["failures"])


# ── rule 4: implausible ratio ────────────────────────────────────────────────

def test_implausible_occupancy_fails(implausible_occupancy_df):
    result = run_validation(implausible_occupancy_df, use_ge=False)
    assert result["success"] is False
    assert any("1.5" in f for f in result["failures"])


def test_borderline_occupancy_passes(good_df):
    # 149 / 100 = 1.49 — just under the 1.5× threshold
    df = good_df.copy()
    df.loc[0, "icu_occupied"] = 149
    df.loc[0, "icu_capacity"] = 100
    result = run_validation(df, use_ge=False)
    assert result["success"] is True


# ── rule 5: zero / negative capacity ─────────────────────────────────────────

def test_zero_capacity_fails(zero_capacity_df):
    result = run_validation(zero_capacity_df, use_ge=False)
    assert result["success"] is False
    assert any("zero or negative icu_capacity" in f for f in result["failures"])


def test_negative_capacity_fails(good_df):
    df = good_df.copy()
    df.loc[0, "icu_capacity"] = -10
    result = run_validation(df, use_ge=False)
    assert result["success"] is False
    assert any("zero or negative icu_capacity" in f for f in result["failures"])
