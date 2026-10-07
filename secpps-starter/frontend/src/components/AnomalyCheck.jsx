import { useEffect, useState } from "react";
import { dashboardApi } from "../api/dashboardApi";
import AnomalyList from "./AnomalyList";
import AnomalyChart from "./AnomalyChart";
import Warnings from "./Warnings";

const label = (ts) => ts.replace("T", " ").slice(0, 16);

function useSeconds(active) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!active) return undefined;
    setSeconds(0);
    const id = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, [active]);
  return seconds;
}

/**
 * "Check for anomalies": asks the backend to run the team's saved anomaly models over the user's own
 * readings, then shows how many unusual events were found, when, how unusual, and where they sit.
 */
export default function AnomalyCheck({ token }) {
  const [status, setStatus] = useState("idle"); // idle | checking | done | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const seconds = useSeconds(status === "checking");

  const handleCheck = async (refresh = false) => {
    setStatus("checking");
    setError(null);
    try {
      setResult(await dashboardApi.getMyAnomalies(token, { refresh }));
      setStatus("done");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  };

  const summary = result && result.summary;
  const found = result ? result.anomaly_count : 0;
  const low = summary ? Math.max(found - summary.high_severity - summary.medium_severity, 0) : 0;

  return (
    <div className="chart-panel">
      <div className="anomaly-check-header">
        <h2>Anomaly check</h2>
        <button onClick={() => handleCheck(status === "done")} disabled={status === "checking"}>
          {status === "checking" ? "Checking..." : status === "done" ? "Check again" : "Check for anomalies"}
        </button>
      </div>

      {status === "idle" && (
        <p className="hint">
          Looks through your readings for unusual consumption: sudden spikes, drop-outs and odd patterns.
        </p>
      )}

      {status === "checking" && (
        <div className="progress-now" role="status">
          <span className="spinner" aria-hidden="true" />
          <span>
            Analysing your readings ({seconds}s). The first check also loads the models, so it can take up to
            a minute.
          </span>
        </div>
      )}

      {status === "error" && (
        <div className="status error" role="alert">
          <p>{error}</p>
        </div>
      )}

      {status === "done" && result && (
        <>
          {result.status !== "ok" ? (
            <p className="status">{result.message}</p>
          ) : (
            <>
              {found === 0 ? (
                <p className="anomaly-clean">✓ No unusual consumption events were found in your readings.</p>
              ) : (
                <p className="anomaly-headline">
                  <strong>{found.toLocaleString()}</strong> unusual {found === 1 ? "event" : "events"} found
                  <span className="anomaly-breakdown">
                    {" "}({summary.high_severity} high, {summary.medium_severity} medium, {low} low)
                  </span>
                </p>
              )}

              <p className="hint">
                Checked {summary.readings_scored.toLocaleString()} hourly readings from{" "}
                {label(summary.history_start)} to {label(summary.history_end)}.
                {result.latest &&
                  ` Latest reading (${label(result.latest.timestamp)}): ${
                    result.latest.is_anomaly ? `unusual, ${result.latest.severity.toLowerCase()}` : "normal"
                  }.`}
              </p>

              <Warnings items={result.warnings} />

              <AnomalyChart readings={result.recent_readings} />
              {found > 0 && <AnomalyList anomalies={result.anomalies} total={found} />}

              <p className="hint anomaly-footnote">
                Scores run from 0 (normal) to 1 (very unusual); they are not probabilities. A reading is flagged
                by the team's saved Isolation Forest + LSTM autoencoder ensemble.
                {result.source === "upload" &&
                  " Weather for this check assumes the Newcastle, NSW area, because the detector has no location input yet."}
              </p>
            </>
          )}
        </>
      )}
    </div>
  );
}