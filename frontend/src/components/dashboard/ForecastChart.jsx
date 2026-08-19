import {
    LineChart,
    Line,
    Area,
    ComposedChart,
    XAxis,
    YAxis,
    Tooltip,
    CartesianGrid,
    ResponsiveContainer,
    ReferenceLine,
    Legend,
  } from "recharts";
  import EmptyState from "../common/EmptyState";
  
  export default function ForecastChart({ forecasts, capacity }) {
    if (!forecasts || forecasts.length === 0) {
      return <EmptyState message="No forecast data available." />;
    }
  
    const sortedForecasts = [...forecasts].sort(
      (a, b) => new Date(a.forecast_time) - new Date(b.forecast_time)
    );

    // Detect model name from the first forecast that has one
    const modelName = sortedForecasts.find((f) => f.model_name)?.model_name || "unknown";
  
    const chartData = sortedForecasts.map((item, idx) => ({
      time: `Week ${idx + 1}`,
      predicted: Number(Number(item.predicted_icu_occupied).toFixed(2)),
      yhat_lower: item.yhat_lower != null ? Number(Number(item.yhat_lower).toFixed(2)) : null,
      yhat_upper: item.yhat_upper != null ? Number(Number(item.yhat_upper).toFixed(2)) : null,
      // recharts Area needs a range array for the band
      ci: item.yhat_lower != null && item.yhat_upper != null
        ? [Number(Number(item.yhat_lower).toFixed(2)), Number(Number(item.yhat_upper).toFixed(2))]
        : null,
    }));

    const hasCI = chartData.some((d) => d.ci !== null);
  
    return (
      <div className="card chart-card">
        <div className="section-title">
          Forecasted ICU Occupancy
          <span style={{ fontSize: "0.75rem", fontWeight: 400, marginLeft: "0.75rem", color: "#6b7280" }}>
            Model: <strong style={{ color: "#2563eb" }}>{modelName}</strong>
          </span>
        </div>
  
        <div className="chart-wrapper">
          <ResponsiveContainer width="100%" height={320}>
            <ComposedChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip
                formatter={(value, name) => {
                  if (name === "ci") {
                    return [`${value[0]} – ${value[1]}`, "95% CI"];
                  }
                  return [value, name === "predicted" ? "Predicted" : name];
                }}
              />
              <Legend />
              <ReferenceLine
                y={capacity * 0.75}
                label={{ value: "Yellow", position: "right", fill: "#b45309", fontSize: 12 }}
                stroke="#f59e0b"
                strokeDasharray="5 5"
              />
              <ReferenceLine
                y={capacity * 0.9}
                label={{ value: "Red", position: "right", fill: "#dc2626", fontSize: 12 }}
                stroke="#ef4444"
                strokeDasharray="5 5"
              />
              {hasCI && (
                <Area
                  dataKey="ci"
                  fill="#2563eb"
                  fillOpacity={0.12}
                  stroke="none"
                  name="95% CI"
                  legendType="rect"
                  isAnimationActive={false}
                />
              )}
              <Line
                type="monotone"
                dataKey="predicted"
                stroke="#2563eb"
                strokeWidth={2.5}
                dot={{ r: 4, fill: "#2563eb" }}
                name="Predicted"
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
  
        <div className="chart-note">
          Forecast uses the <strong>{modelName}</strong> model with Monte Carlo 95% confidence intervals.
        </div>
      </div>
    );
  }