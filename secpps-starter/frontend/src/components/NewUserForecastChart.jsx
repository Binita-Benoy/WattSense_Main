import { useMemo } from "react";
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

// How much of the user's past to show next to each forecast
const HISTORY_DAYS = { "24h": 7, "7day": 30, "1month": 90 };
const round = (n) => Math.round(n * 1000) / 1000;

function dailyTotals(history) {
  const byDay = new Map();
  for (const p of history) {
    const day = p.timestamp.slice(0, 10);
    const entry = byDay.get(day) || { kwh: 0, hours: 0 };
    entry.kwh += p.kwh;
    entry.hours += 1;
    byDay.set(day, entry);
  }
  // Only complete days count; a part-day at either end would show up as a false dip.
  return [...byDay.entries()]
    .filter(([, e]) => e.hours === 24)
    .map(([day, e]) => ({ ts: day, value: round(e.kwh) }));
}

/**
 * Builds the chart rows. The 24-hour forecast is hourly, so history is shown hourly too.
 * The 7-day and 1-month forecasts are daily totals, so history is summed per day: putting
 * hourly kWh and daily kWh on one axis would be meaningless.
 */
export function buildSeries(history, horizonData, horizon) {
  const hourly = horizon === "24h";
  const keep = HISTORY_DAYS[horizon];
  const past = hourly
    ? history.slice(-keep * 24).map((p) => ({ ts: p.timestamp, value: p.kwh }))
    : dailyTotals(history).slice(-keep);
  const future = horizonData.predictions.map((p) => ({
    ts: hourly ? p.timestamp : p.date,
    value: p.predicted_kwh,
  }));

  const rows = past.map((p) => ({ ts: p.ts, history: p.value }));
  const last = past[past.length - 1];
  // Give the forecast line a starting point on the last history point so the lines join up.
  if (last) rows[rows.length - 1].forecast = last.value;
  future.forEach((p) => rows.push({ ts: p.ts, forecast: p.value }));

  return { rows, lastHistoryTs: last ? last.ts : null, hourly };
}

function formatTick(ts, hourly) {
  const dayMonth = `${ts.slice(8, 10)}/${ts.slice(5, 7)}`;
  return hourly ? `${dayMonth} ${ts.slice(11, 13)}:00` : dayMonth;
}

export default function NewUserForecastChart({ history, horizon, horizonData }) {
  const series = useMemo(
    () => (horizonData && !horizonData.error ? buildSeries(history, horizonData, horizon) : null),
    [history, horizon, horizonData]
  );

  if (!horizonData) return null;

  if (horizonData.error) {
    return (
      <div className="chart-panel">
        <h2>Consumption &amp; forecast</h2>
        <div className="status error">{horizonData.error}</div>
      </div>
    );
  }

  return (
    <div className="chart-panel">
      <h2>
        Consumption &amp; forecast{" "}
        <span className="chart-unit">({series.hourly ? "kWh per hour" : "kWh per day"})</span>
      </h2>
      <ResponsiveContainer width="100%" height={340}>
        <ComposedChart data={series.rows} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#232a35" strokeDasharray="3 3" />
          <XAxis
            dataKey="ts"
            tickFormatter={(ts) => formatTick(ts, series.hourly)}
            stroke="#8993a4"
            fontSize={12}
            minTickGap={28}
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
            labelFormatter={(ts) => ts.replace("T", " ").slice(0, series.hourly ? 16 : 10)}
            formatter={(value, name) => [`${Number(value).toFixed(3)} kWh`, name]}
          />
          <Legend />
          {series.lastHistoryTs && (
            <ReferenceLine
              x={series.lastHistoryTs}
              stroke="#8993a4"
              strokeDasharray="3 3"
              label={{ value: "Forecast starts", fill: "#8993a4", fontSize: 11, position: "insideTopRight" }}
            />
          )}
          <Line
            type="monotone"
            dataKey="history"
            stroke="#5eead4"
            strokeWidth={2}
            dot={false}
            name="Your usage"
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="forecast"
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