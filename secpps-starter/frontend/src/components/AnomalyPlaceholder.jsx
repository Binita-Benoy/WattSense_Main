
export default function AnomalyPlaceholder() {
  return (
    <div className="chart-panel anomaly-placeholder">
      <div className="anomaly-check-header">
        <h2>Anomaly check</h2>
        <button disabled title="Not available for uploaded data yet">
          Check for anomalies
        </button>
      </div>
      <p className="hint">
        Unusual-consumption checks for uploaded data aren't available yet. Once the anomaly model is
        connected, flagged hours (time, usage and severity) will appear here.
      </p>
    </div>
  );
}