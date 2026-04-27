import KpiCard from "./KpiCard";
import { formatNumber, formatPercent } from "../../utils/format";
import { computeOccupancyRatio } from "../../utils/risk";

export default function KpiGrid({ snapshot, latestForecast, forecasts }) {
  if (!snapshot) return null;

  const occupancyRatio = computeOccupancyRatio(
    snapshot.icu_occupied,
    snapshot.icu_capacity
  );

  const availableBeds = snapshot.icu_capacity - snapshot.icu_occupied;

  // Determine model name and max breach probability from forecasts
  const modelName = latestForecast?.model_name || "--";
  const maxBreachProb = (forecasts && forecasts.length > 0)
    ? Math.max(...forecasts.map((f) => f.breach_prob ?? 0))
    : null;

  return (
    <div className="kpi-grid">
      <KpiCard
        label="ICU Capacity"
        value={formatNumber(snapshot.icu_capacity)}
      />
      <KpiCard
        label="ICU Occupied"
        value={formatNumber(snapshot.icu_occupied)}
      />
      <KpiCard
        label="Available Beds"
        value={formatNumber(availableBeds)}
      />
      <KpiCard
        label="Occupancy Rate"
        value={formatPercent(occupancyRatio)}
      />
      <KpiCard
        label="Admissions"
        value={formatNumber(snapshot.admissions)}
      />
      <KpiCard
        label="Discharges"
        value={formatNumber(snapshot.discharges)}
      />
      <KpiCard
        label="ED Arrivals"
        value={formatNumber(snapshot.ed_arrivals)}
      />
      <KpiCard
        label="Staffing Level"
        value={formatPercent(snapshot.staffing_level)}
      />
      <KpiCard
        label="Next Forecasted Occupancy"
        value={latestForecast ? formatNumber(latestForecast.predicted_icu_occupied.toFixed(1)) : "--"}
      />
      <KpiCard
        label="Forecast Model"
        value={modelName}
        subtext={modelName !== "--" ? "Auto-selected" : undefined}
      />
      <KpiCard
        label="Max Breach Prob"
        value={maxBreachProb !== null ? `${(maxBreachProb * 100).toFixed(1)}%` : "--"}
        subtext="P(occupancy > RED threshold)"
      />
    </div>
  );
}