export default function Topbar({
    hospitalId,
    onRefresh,
    refreshing,
    lastUpdated,
  }) {
    return (
      <header className="topbar">
        <div>
          <div className="topbar-title">Hospital Capacity Dashboard</div>
          <div className="topbar-subtitle">
            Monitoring current ICU load and short-term forecast
          </div>
          <div className="topbar-last-updated">
            Last updated: {lastUpdated || "--"}
          </div>
        </div>
  
        <div className="topbar-actions">
          <div className="topbar-hospital">{hospitalId}</div>
          <button className="primary-btn" onClick={onRefresh} disabled={refreshing}>
            {refreshing ? "Refreshing..." : "Run Forecast + Refresh"}
          </button>
        </div>
      </header>
    );
  }