"""Locust load test for the hospital ICU forecasting API.

Simulates 1 user per hospital hitting all four endpoints with realistic
task weights. Hospital IDs are assigned round-robin via user index so
each user owns exactly one hospital for the duration of the run.

Usage:
    locust -f tests/load/locustfile.py \\
      --host http://52.15.187.251:8000 \\
      --users 10 --spawn-rate 1 \\
      --run-time 120s --headless \\
      --csv reports/run_$(date +%Y%m%d_%H%M%S)
"""
from __future__ import annotations

from locust import HttpUser, between, events, task

from tests.load.mlflow_reporter import log_run

HOSPITAL_IDS = [f"H{str(i).zfill(3)}" for i in range(1, 11)]  # H001–H010
_user_counter = 0


class HospitalUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        global _user_counter
        self.hospital_id = HOSPITAL_IDS[_user_counter % len(HOSPITAL_IDS)]
        _user_counter += 1

    @task(5)
    def get_latest_snapshot(self):
        self.client.get(
            f"/snapshots/latest/{self.hospital_id}",
            name="/snapshots/latest/[id]",
        )

    @task(4)
    def get_forecasts(self):
        self.client.get(
            f"/forecasts/{self.hospital_id}",
            name="/forecasts/[id]",
        )

    @task(3)
    def get_alerts(self):
        self.client.get(
            f"/alerts/{self.hospital_id}",
            name="/alerts/[id]",
        )

    @task(1)
    def run_forecast(self):
        self.client.post(
            f"/forecast/{self.hospital_id}",
            name="/forecast/[id]",
        )


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    log_run(environment)
