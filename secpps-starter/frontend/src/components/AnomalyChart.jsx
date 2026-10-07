import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const SEVERITY_COLORS = { HIGH: "#ef4444", MEDIUM: "#f59e0b", LOW: "#eab308" };

// Timestamps are plain local times ("2014-02-28T21:00:00"); slice them rather than parse them,
// so the browser's timezone can never shift them.
const tick = (ts) => `${ts.slice(8, 10)}/${ts.slice(5, 7)} ${ts.slice(11, 13)}:00`;
const label = (ts) => ts.replace("T", " ").slice(0, 16);

function FlagDot({ cx, cy, payload }) {
  if (payload == null || payload.flagged == null || cx == null || cy == null) return null;
  return (
    <circle
      cx={cx}
      cy={cy}
      r={5}
      fill={SEVERITY_COLORS[payload.severity] || "#ef4444"}
      stroke="#0b0f14"
      strokeWidth={1}
    />
  );
}

function ChartTooltip({ active, payload }) {
  if (!active || !payload || payload.length === 0) return null;
  const row = payload[0].payload;
  return (
    <div className="anomaly-tooltip">
      <div>{label(row.ts)}</div>
      <div>{Number(row.consumption).toFixed(3)} kWh</div>
      {row.flagged != null && (
        <div style={{ color: SEVERITY_COLORS[row.severity] || "#ef4444" }}>
          Unusual: {row.severity}, score {Number(row.score).toFixed(2)}
        </div>
      )}
    </div>
  );
}

/** The most recent readings as a line, with the unusual ones marked (colour = severity). */
export default function AnomalyChart({ readings }) {
  if (!readings || readings.length === 0) return null;

  const rows = readings.map((r) => ({
    ts: r.timestamp,
    consumption: r.consumption,
    flagged: r.is_anomaly ? r.consumption : null,
    severity: r.severity,
    score: r.anomaly_score,
  }));
  const flaggedCount = rows.filter((r) => r.flagged != null).length;

  return (
    <div className="anomaly-chart">
      <h3>
        Latest {Math.round(readings.length / 24)} days of readings{" "}
        <span className="chart-unit">
          ({flaggedCount === 0 ? "nothing unusual in this stretch" : `${flaggedCount} unusual, marked`})
        </span>
      </h3>
      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#232a35" strokeDasharray="3 3" />
          <XAxis dataKey="ts" tickFormatter={tick} stroke="#8993a4" fontSize={12} minTickGap={32} />
          <YAxis stroke="#8993a4" fontSize={12} unit=" kWh" />
          <Tooltip content={<ChartTooltip />} />
          <Line type="monotone" dataKey="consumption" stroke="#5eead4" strokeWidth={2} dot={false} isAnimationActive={false} />
          <Line
            dataKey="flagged"
            stroke="none"
            dot={<FlagDot />}
            activeDot={false}
            connectNulls={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="anomaly-legend">
        <span><i style={{ background: "#5eead4" }} /> Usage</span>
        <span><i className="dot" style={{ background: SEVERITY_COLORS.HIGH }} /> High</span>
        <span><i className="dot" style={{ background: SEVERITY_COLORS.MEDIUM }} /> Medium</span>
        <span><i className="dot" style={{ background: SEVERITY_COLORS.LOW }} /> Low</span>
      </div>
    </div>
  );
}