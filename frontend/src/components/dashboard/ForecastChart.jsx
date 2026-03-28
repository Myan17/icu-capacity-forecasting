import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    Tooltip,
    CartesianGrid,
    ResponsiveContainer,
    ReferenceLine,
  } from "recharts";
  import EmptyState from "../common/EmptyState";
  
  export default function ForecastChart({ forecasts, capacity }) {
    if (!forecasts || forecasts.length === 0) {
      return <EmptyState message="No forecast data available." />;
    }
  
    const sortedForecasts = [...forecasts].sort(
      (a, b) => new Date(a.forecast_time) - new Date(b.forecast_time)
    );
  
    const chartData = sortedForecasts.map((item) => ({
      time: new Date(item.forecast_time).toLocaleTimeString(),
      predicted: Number(item.predicted_icu_occupied.toFixed(2)),
    }));
  
    return (
      <div className="card chart-card">
        <div className="section-title">Forecasted ICU Occupancy</div>
  
        <div className="chart-wrapper">
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <ReferenceLine y={capacity * 0.75} label="Yellow Threshold" />
              <ReferenceLine y={capacity * 0.9} label="Red Threshold" />
              <Line type="monotone" dataKey="predicted" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
  
        <div className="chart-note">
          Forecast points are based on the current backend prediction model.
        </div>
      </div>
    );
  }