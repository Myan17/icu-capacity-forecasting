import EmptyState from "../common/EmptyState";
import { formatDateTime } from "../../utils/format";
import { getRiskColor } from "../../utils/risk";

export default function AlertsPanel({ alerts }) {
  const visibleAlerts = alerts ? alerts.slice(0, 10) : [];

  return (
    <div className="card alerts-card">
      <div className="section-title">Active Alerts</div>

      {!visibleAlerts || visibleAlerts.length === 0 ? (
        <EmptyState message="No alerts found." />
      ) : (
        <div className="alerts-list">
          {visibleAlerts.map((alert) => (
            <div key={alert.id} className="alert-item">
              <div className={`alert-badge ${getRiskColor(alert.risk_level)}`}>
                {alert.risk_level}
              </div>
              <div className="alert-content">
                <div className="alert-message">{alert.message}</div>
                <div className="alert-meta">
                  {alert.hospital_id} • {formatDateTime(alert.created_at)}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}