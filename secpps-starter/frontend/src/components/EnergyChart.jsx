import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

function formatTick(ts) {
  const d = new Date(ts);
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

export default function EnergyChart({ readings }) {
  return (
    <div className="chart-panel">
      <h2>Consumption over time</h2>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={readings} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#232a35" strokeDasharray="3 3" />
          <XAxis
            dataKey="reading_timestamp"
            tickFormatter={formatTick}
            stroke="#8993a4"
            fontSize={12}
          />
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
          <Line
            type="monotone"
            dataKey="consumption_kwh"
            stroke="#5eead4"
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}