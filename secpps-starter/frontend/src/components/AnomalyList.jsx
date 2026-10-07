import { useState } from "react";

const SEVERITY_COLORS = { HIGH: "#ef4444", MEDIUM: "#f59e0b", LOW: "#eab308" };
const ORDER = { HIGH: 0, MEDIUM: 1, LOW: 2 };
const EVIDENCE = {
  both: "both models flagged it",
  if_only: "Isolation Forest flagged it",
  ae_only: "LSTM autoencoder flagged it",
  weighted_combination: "flagged by the combined score",
};
const SHOW_FIRST = 15;

const label = (ts) => ts.replace("T", " ").slice(0, 16);

/** Unusual readings, worst first (then newest first). `total` = how many the check found overall. */
export default function AnomalyList({ anomalies, total, message }) {
  const [showAll, setShowAll] = useState(false);

  if (message) {
    return (
      <div className="chart-panel">
        <h2>Anomalies</h2>
        <p className="status">{message}</p>
      </div>
    );
  }

  const sorted = [...anomalies].sort(
    (a, b) => (ORDER[a.severity] ?? 3) - (ORDER[b.severity] ?? 3) || b.timestamp.localeCompare(a.timestamp)
  );
  const visible = showAll ? sorted : sorted.slice(0, SHOW_FIRST);
  const found = total ?? anomalies.length;

  return (
    <div className="anomaly-list-block">
      <h3>Unusual readings ({found})</h3>
      {found > anomalies.length && (
        <p className="hint">Showing the most recent {anomalies.length} of {found}.</p>
      )}
      <div className="anomaly-list">
        {visible.map((a) => (
          <div key={a.timestamp} className="anomaly-row">
            <span className="anomaly-severity" style={{ background: SEVERITY_COLORS[a.severity] || "#64748b" }}>
              {a.severity}
            </span>
            <span className="anomaly-timestamp">{label(a.timestamp)}</span>
            <span className="anomaly-consumption">{Number(a.consumption).toFixed(3)} kWh</span>
            <span className="anomaly-score">score {Number(a.anomaly_score).toFixed(2)}</span>
            <span className="anomaly-evidence">{EVIDENCE[a.model_evidence] || ""}</span>
          </div>
        ))}
      </div>
      {sorted.length > SHOW_FIRST && (
        <button className="link-button anomaly-more" onClick={() => setShowAll((v) => !v)}>
          {showAll ? "Show fewer" : `Show all ${sorted.length}`}
        </button>
      )}
    </div>
  );
}