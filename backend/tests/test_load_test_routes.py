"""Tests for /load-test/latest and /load-test/runs backend routes."""
from __future__ import annotations

import mlflow
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_load_test_latest_returns_404_when_no_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    response = client.get("/load-test/latest")
    assert response.status_code == 404


def test_load_test_latest_returns_hospital_breakdown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    mlflow.set_experiment("hospital-load-test")
    with mlflow.start_run():
        mlflow.log_metrics({"010001_p95_ms": 120.0, "010001_failure_rate": 0.01,
                             "010001_p50_ms": 60.0, "010001_p99_ms": 200.0, "010001_rps": 2.5,
                             "overall_p95_ms": 120.0, "overall_failure_rate": 0.01,
                             "overall_rps": 2.5})
        mlflow.set_tags({"num_hospitals": "1", "run_duration_s": "30", "host": "test"})

    response = client.get("/load-test/latest")
    assert response.status_code == 200
    body = response.json()
    assert "hospitals" in body
    assert len(body["hospitals"]) == 1
    assert body["hospitals"][0]["hospital_id"] == "010001"
    assert body["hospitals"][0]["p95_ms"] == 120.0


def test_load_test_runs_returns_list(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    mlflow.set_experiment("hospital-load-test")
    for _ in range(2):
        with mlflow.start_run():
            mlflow.log_metrics({"overall_p95_ms": 100.0, "overall_failure_rate": 0.0,
                                 "overall_rps": 3.0})
            mlflow.set_tags({"num_hospitals": "2", "run_duration_s": "30"})

    response = client.get("/load-test/runs")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 2
    assert "overall_p95_ms" in body[0]
