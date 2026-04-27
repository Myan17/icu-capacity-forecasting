import { useEffect, useState } from "react";
import { getAllAlerts } from "../api/hospitalApi";
import AppShell from "../components/layout/AppShell";
import LoadingState from "../components/common/LoadingState";

const RISK_COLORS = { RED: "#ef4444", YELLOW: "#f59e0b", GREEN: "#22c55e" };

export default function AlertsPage() {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getAllAlerts(100)
      .then(setAlerts)
      .catch(() => setAlerts([]))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <AppShell><LoadingState message="Loading alerts..." /></AppShell>;

  return (
    <AppShell>
      <div className="card">
        <h1>Alerts</h1>
        <p style={{ color: "#6b7280", marginBottom: "1.5rem" }}>
          {alerts.length} active alert{alerts.length !== 1 ? "s" : ""} across all hospitals
        </p>

        {alerts.length === 0 ? (
          <p style={{ color: "#6b7280" }}>No alerts. All hospitals are within safe thresholds.</p>
        ) : (
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.875rem" }}>
            <thead>
              <tr style={{ background: "#f3f4f6" }}>
                {["Hospital", "Risk", "Message", "Time"].map((h) => (
                  <th key={h} style={{ padding: "0.5rem 1rem", textAlign: "left", borderBottom: "1px solid #e5e7eb" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {alerts.map((a) => (
                <tr key={a.id} style={{ borderBottom: "1px solid #f3f4f6" }}>
                  <td style={{ padding: "0.5rem 1rem", fontWeight: 600 }}>{a.hospital_id}</td>
                  <td style={{ padding: "0.5rem 1rem" }}>
                    <span style={{
                      background: RISK_COLORS[a.risk_level] || "#6b7280",
                      color: "#fff",
                      borderRadius: "4px",
                      padding: "0.15rem 0.5rem",
                      fontSize: "0.75rem",
                      fontWeight: 700,
                    }}>{a.risk_level}</span>
                  </td>
                  <td style={{ padding: "0.5rem 1rem", color: "#374151" }}>{a.message}</td>
                  <td style={{ padding: "0.5rem 1rem", color: "#6b7280", whiteSpace: "nowrap" }}>
                    {a.created_at ? new Date(a.created_at).toLocaleString() : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </AppShell>
  );
}
