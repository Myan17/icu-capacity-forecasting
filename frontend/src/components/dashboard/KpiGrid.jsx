import KpiCard from "./KpiCard";
import { formatNumber, formatPercent } from "../../utils/format";
import { computeOccupancyRatio } from "../../utils/risk";

export default function KpiGrid({ snapshot, latestForecast }) {
  if (!snapshot) return null;

  const occupancyRatio = computeOccupancyRatio(
    snapshot.icu_occupied,
    snapshot.icu_capacity
  );

  const availableBeds = snapshot.icu_capacity - snapshot.icu_occupied;

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
    </div>
  );
}