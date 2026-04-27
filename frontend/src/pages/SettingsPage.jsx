import AppShell from "../components/layout/AppShell";

const SETTINGS = [
  { label: "Yellow Threshold", value: "75%", desc: "ICU occupancy level that triggers a YELLOW alert" },
  { label: "Red Threshold", value: "90%", desc: "ICU occupancy level that triggers a RED alert" },
  { label: "Forecast Model", value: "Auto (ML selector)", desc: "Chooses best of Baseline, SARIMA, Prophet per hospital" },
  { label: "Forecast Horizon", value: "8 weeks", desc: "How far ahead the forecast projects" },
  { label: "Forecast Frequency", value: "Weekly", desc: "Time step between forecast points" },
  { label: "Monte Carlo Samples", value: "500", desc: "Samples used for confidence interval estimation" },
];

export default function SettingsPage() {
  return (
    <AppShell>
      <div className="card">
        <h1>Settings</h1>
        <p style={{ color: "#6b7280", marginBottom: "1.5rem" }}>
          Current system configuration. Edit via <code>.env</code> on the backend server.
        </p>

        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.875rem" }}>
          <thead>
            <tr style={{ background: "#f3f4f6" }}>
              {["Setting", "Value", "Description"].map((h) => (
                <th key={h} style={{ padding: "0.5rem 1rem", textAlign: "left", borderBottom: "1px solid #e5e7eb" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {SETTINGS.map(({ label, value, desc }) => (
              <tr key={label} style={{ borderBottom: "1px solid #f3f4f6" }}>
                <td style={{ padding: "0.5rem 1rem", fontWeight: 600 }}>{label}</td>
                <td style={{ padding: "0.5rem 1rem" }}>
                  <span style={{
                    background: "#eff6ff",
                    color: "#2563eb",
                    borderRadius: "4px",
                    padding: "0.15rem 0.5rem",
                    fontWeight: 600,
                  }}>{value}</span>
                </td>
                <td style={{ padding: "0.5rem 1rem", color: "#6b7280" }}>{desc}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </AppShell>
  );
}
