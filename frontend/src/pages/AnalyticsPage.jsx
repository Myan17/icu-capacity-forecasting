import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, LineChart, Line, Legend,
} from "recharts";
import { getLoadTestLatest, getLoadTestRuns } from "../api/hospitalApi";
import AppShell from "../components/layout/AppShell";
import LoadingState from "../components/common/LoadingState";
import ErrorState from "../components/common/ErrorState";

const TABS = ["Latest Run", "Run History"];

export default function AnalyticsPage() {
  const [tab, setTab] = useState(0);
  const [latest, setLatest] = useState(null);
  const [runs, setRuns]     = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]   = useState("");

  useEffect(() => {
    async function load() {
      try {
        const [lat, hist] = await Promise.allSettled([
          getLoadTestLatest(),
          getLoadTestRuns(),
        ]);
        if (lat.status === "fulfilled") setLatest(lat.value);
        if (hist.status === "fulfilled") setRuns(hist.value);
      } catch {
        setError("Could not load load-test data. Run a Locust test first.");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) return <AppShell><LoadingState message="Loading analytics..." /></AppShell>;

  return (
    <AppShell>
      <div className="card">
        <h1>Load Test Analytics</h1>

        {/* Tab bar */}
        <div style={{ display: "flex", gap: "1rem", marginBottom: "1.5rem" }}>
          {TABS.map((label, i) => (
            <button
              key={label}
              onClick={() => setTab(i)}
              style={{
                padding: "0.4rem 1rem",
                borderRadius: "6px",
                border: "none",
                cursor: "pointer",
                background: tab === i ? "#2563eb" : "#e5e7eb",
                color: tab === i ? "#fff" : "#374151",
                fontWeight: tab === i ? 600 : 400,
              }}
            >
              {label}
            </button>
          ))}
        </div>

        {error && <ErrorState message={error} />}

        {/* Tab 1 — Latest Run */}
        {tab === 0 && (
          latest ? (
            <>
              {/* Stat cards */}
              <div style={{ display: "flex", gap: "1rem", marginBottom: "1.5rem", flexWrap: "wrap" }}>
                {[
                  { label: "Overall p95 (ms)", value: latest.overall_p95_ms },
                  { label: "Overall RPS",       value: latest.overall_rps },
                  { label: "Failure Rate",       value: `${(latest.overall_failure_rate * 100).toFixed(2)}%` },
                  { label: "Hospitals Tested",   value: latest.num_hospitals },
                  { label: "Run Duration (s)",   value: latest.run_duration_s },
                ].map(({ label, value }) => (
                  <div key={label} className="card" style={{ minWidth: "140px", textAlign: "center" }}>
                    <div style={{ fontSize: "0.75rem", color: "#6b7280" }}>{label}</div>
                    <div style={{ fontSize: "1.5rem", fontWeight: 700 }}>{value}</div>
                  </div>
                ))}
              </div>

              {/* p95 bar chart per hospital */}
              <div className="section-title">p95 Latency per Hospital (ms)</div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={latest.hospitals} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hospital_id" />
                  <YAxis unit=" ms" />
                  <Tooltip formatter={(v) => `${v} ms`} />
                  <Bar dataKey="p95_ms" fill="#2563eb" name="p95 latency" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>

              <div style={{ fontSize: "0.75rem", color: "#6b7280", marginTop: "0.5rem" }}>
                Run: {latest.timestamp ? new Date(latest.timestamp).toLocaleString() : "—"}
              </div>
            </>
          ) : (
            <p style={{ color: "#6b7280" }}>
              No load test data yet. Run: <code>locust -f tests/load/locustfile.py --host http://52.15.187.251:8000 --users 5 --spawn-rate 1 --run-time 60s --headless</code>
            </p>
          )
        )}

        {/* Tab 2 — Run History */}
        {tab === 1 && (
          runs.length > 0 ? (
            <>
              <div className="section-title">p95 Latency Trend Across Runs</div>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart
                  data={[...runs].reverse().map((r) => ({
                    ...r,
                    label: r.timestamp
                      ? new Date(r.timestamp).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
                      : r.run_id.slice(0, 8),
                  }))}
                  margin={{ top: 10, right: 20, left: 0, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                  <YAxis unit=" ms" />
                  <Tooltip formatter={(v) => `${v} ms`} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="overall_p95_ms"
                    stroke="#2563eb"
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    name="Overall p95 (ms)"
                  />
                </LineChart>
              </ResponsiveContainer>

              {/* History table */}
              <table style={{ width: "100%", borderCollapse: "collapse", marginTop: "1.5rem", fontSize: "0.875rem" }}>
                <thead>
                  <tr style={{ background: "#f3f4f6" }}>
                    {["Timestamp", "Hospitals", "p95 (ms)", "Failure Rate", "RPS", "Duration (s)"].map(h => (
                      <th key={h} style={{ padding: "0.5rem 1rem", textAlign: "left", borderBottom: "1px solid #e5e7eb" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {runs.map((r) => (
                    <tr key={r.run_id} style={{ borderBottom: "1px solid #f3f4f6" }}>
                      <td style={{ padding: "0.5rem 1rem" }}>{r.timestamp ? new Date(r.timestamp).toLocaleString() : "—"}</td>
                      <td style={{ padding: "0.5rem 1rem" }}>{r.num_hospitals}</td>
                      <td style={{ padding: "0.5rem 1rem" }}>{r.overall_p95_ms}</td>
                      <td style={{ padding: "0.5rem 1rem" }}>{(r.overall_failure_rate * 100).toFixed(2)}%</td>
                      <td style={{ padding: "0.5rem 1rem" }}>{r.overall_rps}</td>
                      <td style={{ padding: "0.5rem 1rem" }}>{r.run_duration_s}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          ) : (
            <p style={{ color: "#6b7280" }}>No run history yet. Complete at least one load test to see the trend.</p>
          )
        )}
      </div>
    </AppShell>
  );
}
