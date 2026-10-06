const SEVERITY_COLORS = {
  HIGH: "#ef4444",
  MEDIUM: "#f59e0b",
  LOW: "#eab308",
};

export default function AnomalyList({ anomalies, message }) {
  if (message) {
    return (
      <div className="chart-panel">
        <h2>Anomalies</h2>
        <p className="status">{message}</p>
      </div>
    );
  }

  return (
    <div className="chart-panel">
      <h2>Anomalies ({anomalies.length})</h2>
      <div className="anomaly-list">
        {anomalies.map((a, i) => (
          <div key={i} className="anomaly-row">
            <span
              className="anomaly-severity"
              style={{ background: SEVERITY_COLORS[a.severity] || "#64748b" }}
            >
              {a.severity}
            </span>
            <span className="anomaly-timestamp">
              {new Date(a.timestamp).toLocaleString()}
            </span>
            <span className="anomaly-consumption">{a.consumption} kWh</span>
            <span className="anomaly-score">score: {a.anomaly_score.toFixed(3)}</span>
            <span className="anomaly-detector">{a.detector}</span>
          </div>
        ))}
      </div>
    </div>
  );
}