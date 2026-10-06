import { useState } from "react";
import { dashboardApi } from "../api/dashboardApi";
import AnomalyList from "./AnomalyList";

export default function AnomalyCheck({ token }) {
  const [status, setStatus] = useState("idle"); // idle | checking | done | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleCheck = async () => {
    setStatus("checking");
    setError(null);
    try {
      const data = await dashboardApi.getMyAnomalies(token);
      setResult(data);
      setStatus("done");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  };

  return (
    <div className="chart-panel">
      <div className="anomaly-check-header">
        <h2>Anomaly check</h2>
        <button onClick={handleCheck} disabled={status === "checking"}>
          {status === "checking" ? "Checking..." : "Check for anomalies"}
        </button>
      </div>

      {status === "error" && <div className="status error">{error}</div>}

      {status === "done" && result && (
        <>
          {result.anomaly_count === 0 ? (
            <p className="anomaly-clean">
              ✓ No unusual consumption events detected for your household.
            </p>
          ) : (
            <AnomalyList anomalies={result.anomalies} message={null} />
          )}
        </>
      )}
    </div>
  );
}