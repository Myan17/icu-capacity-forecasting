import { useEffect, useMemo, useState } from "react";
import {
  getAlerts,
  getForecasts,
  getLatestSnapshot,
  runForecast,
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
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState("");

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
   
      await runForecast(selectedHospitalId);
      const [snapshotData, forecastData, alertsData] = await Promise.all([
        getLatestSnapshot(selectedHospitalId),
        getForecasts(selectedHospitalId),
        getAlerts(selectedHospitalId),
      ]);

      setSnapshot(snapshotData);
      setForecasts(forecastData);
      setAlerts(alertsData);
      setLastUpdated(formatDateTime(new Date()));
    } catch (err) {
      setError("Could not load dashboard data. Please verify the backend is running.");
    }
  }

  async function handleRefresh() {
    setRefreshing(true);
    setError("");

    try {
      await runForecast(hospitalId);
      await loadDashboardData(hospitalId);
    } catch (err) {
      setError("Could not refresh forecast data.");
    } finally {
      setRefreshing(false);
    }
  }

  useEffect(() => {
    async function init() {
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

      {loading ? (
        <LoadingState message="Loading dashboard..." />
      ) : error ? (
        <ErrorState message={error} />
      ) : (
        <>
          <StatusBanner riskLevel={currentRisk} hospitalId={hospitalId} />

          <KpiGrid snapshot={snapshot} latestForecast={latestForecast} />

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