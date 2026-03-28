import { getRiskColor } from "../../utils/risk";

export default function StatusBanner({ riskLevel = "UNKNOWN", hospitalId }) {
  return (
    <div className={`status-banner ${getRiskColor(riskLevel)}`}>
      <div className="status-title">Current Risk Status</div>
      <div className="status-main">
        {hospitalId} — {riskLevel}
      </div>
    </div>
  );
}