import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

function formatTick(ts) {
  const d = new Date(ts);
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

/**
 * Historical consumption (solid line) + forecast (dashed line) on one chart.
 * Matches real EnergyReading field names (reading_timestamp, consumption_kwh)
 * and the forecast endpoint's field names (timestamp, predicted_kwh).
 */
export default function ForecastChart({ readings, forecast }) {
  const merged = [
    ...readings.map((r) => ({
      ts: r.reading_timestamp,
      consumption_kwh: r.consumption_kwh,
    })),
    ...forecast.map((f) => ({
      ts: f.timestamp,
      predicted_kwh: f.predicted_kwh,
    })),
  ];

  return (
    <div className="chart-panel">
      <h2>Consumption &amp; forecast</h2>
      <ResponsiveContainer width="100%" height={340}>
        <ComposedChart data={merged} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#232a35" strokeDasharray="3 3" />
          <XAxis dataKey="ts" tickFormatter={formatTick} stroke="#8993a4" fontSize={12} />
          <YAxis stroke="#8993a4" fontSize={12} unit=" kWh" />
          <Tooltip
            contentStyle={{
              background: "#141922",
              border: "1px solid #232a35",
              borderRadius: 6,
              fontFamily: "JetBrains Mono, monospace",
              fontSize: 12,
            }}
            labelFormatter={(ts) => new Date(ts).toLocaleString()}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="consumption_kwh"
            stroke="#5eead4"
            strokeWidth={2}
            dot={false}
            name="Historical"
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="predicted_kwh"
            stroke="#f59e0b"
            strokeWidth={2}
            strokeDasharray="6 4"
            dot={false}
            name="Forecast"
            connectNulls
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}