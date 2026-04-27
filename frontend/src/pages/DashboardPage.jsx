import { useEffect, useMemo, useRef, useState } from "react";
import {
  getAlerts,
  getForecasts,
  getLatestSnapshot,
  pollForNewForecast,
  triggerForecast,
} from "../api/hospitalApi";
import { formatDateTime } from "../utils/format";
import AppShell from "../components/layout/AppShell";
import Topbar from "../components/layout/Topbar";
import HospitalSelector from "../components/dashboard/HospitalSelector";
import StatusBanner from "../components/dashboard/StatusBanner";
import KpiGrid from "../components/dashboard/KpiGrid";
import ForecastChart from "../components/dashboard/ForecastChart";
import AlertsPanel from "../components/dashboard/AlertsPanel";
import LoadingState from "../components/common/LoadingState";
import ErrorState from "../components/common/ErrorState";

export default function DashboardPage() {
  const [hospitalId, setHospitalId] = useState("010001");
  const [snapshot, setSnapshot] = useState(null);
  const [forecasts, setForecasts] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [forecasting, setForecasting] = useState(false);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState("");

  // Incremented whenever we switch hospitals or start a new poll, so stale
  // background polls don't overwrite state after the user has moved on.
  const pollGenRef = useRef(0);

  const latestForecast = useMemo(() => {
    if (!forecasts.length) return null;

    const sortedForecasts = [...forecasts].sort(
      (a, b) => new Date(a.forecast_time) - new Date(b.forecast_time)
    );

    return sortedForecasts[0];
  }, [forecasts]);

  const currentRisk = useMemo(() => {
    if (alerts.length > 0) return alerts[0].risk_level;
    if (latestForecast?.risk_level) return latestForecast.risk_level;
    return "UNKNOWN";
  }, [alerts, latestForecast]);

  async function loadDashboardData(selectedHospitalId) {
    setError("");

    try {
      const [snapshotData, forecastData, alertsData] = await Promise.all([
        getLatestSnapshot(selectedHospitalId),
        getForecasts(selectedHospitalId),
        getAlerts(selectedHospitalId),
      ]);

      setSnapshot(snapshotData);
      setForecasts(forecastData);
      setAlerts(alertsData);
      setLastUpdated(formatDateTime(new Date()));
    } catch {
      setError("Could not load dashboard data. Please verify the backend is running.");
    }
  }

  async function handleRefresh() {
    setRefreshing(true);
    setError("");

    const priorRunId = forecasts[0]?.run_id ?? null;
    const gen = ++pollGenRef.current;

    try {
      await triggerForecast(hospitalId);         // fast 202
      await loadDashboardData(hospitalId);        // show current data immediately
    } catch {
      setError("Could not trigger forecast.");
      setRefreshing(false);
      return;
    }

    setRefreshing(false);
    setForecasting(true);

    // Background poll — updates chart when new run_id arrives
    pollForNewForecast(hospitalId, priorRunId).then((newData) => {
      if (gen !== pollGenRef.current) return; // user switched hospital
      if (newData) {
        setForecasts(newData);
        setLastUpdated(formatDateTime(new Date()));
      }
      setForecasting(false);
    });
  }

  useEffect(() => {
    pollGenRef.current++; // cancel any in-flight poll from a previous hospital

    async function init() {
      setForecasting(false);
      setLoading(true);
      await loadDashboardData(hospitalId);
      setLoading(false);
    }

    init();
  }, [hospitalId]);

  return (
    <AppShell>
      <Topbar
        hospitalId={hospitalId}
        onRefresh={handleRefresh}
        refreshing={refreshing}
        lastUpdated={lastUpdated}
      />

      <div className="page-controls">
        <HospitalSelector value={hospitalId} onChange={setHospitalId} />
      </div>

      {forecasting && (
        <div className="forecast-updating-banner">
          <span className="forecast-updating-spinner" />
          Updating forecast — chart will refresh when results are ready…
        </div>
      )}

      {loading ? (
        <LoadingState message="Loading dashboard..." />
      ) : error ? (
        <ErrorState message={error} />
      ) : (
        <>
          <StatusBanner riskLevel={currentRisk} hospitalId={hospitalId} />

          <KpiGrid snapshot={snapshot} latestForecast={latestForecast} forecasts={forecasts} />

          <div className="dashboard-grid">
            <ForecastChart
              forecasts={forecasts}
              capacity={snapshot?.icu_capacity || 0}
            />
            <AlertsPanel alerts={alerts} />
          </div>
        </>
      )}
    </AppShell>
  );
}