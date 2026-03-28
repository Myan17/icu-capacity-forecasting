export default function KpiCard({ label, value, subtext }) {
    return (
      <div className="card kpi-card">
        <div className="kpi-label">{label}</div>
        <div className="kpi-value">{value}</div>
        {subtext ? <div className="kpi-subtext">{subtext}</div> : null}
      </div>
    );
  }