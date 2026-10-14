import { useEffect, useMemo, useRef, useState } from "react";
import { dashboardApi } from "../api/dashboardApi";
import AnomalyList from "./AnomalyList";
import AnomalyChart from "./AnomalyChart";
import Warnings from "./Warnings";

const label = (ts) => ts.replace("T", " ").slice(0, 16);

// >>> ADDED: helpers for the "click an event, see it on the chart" view (timestamps stay plain text)
const pad = (n) => String(n).padStart(2, "0");
const toMs = (ts) => Date.UTC(+ts.slice(0, 4), +ts.slice(5, 7) - 1, +ts.slice(8, 10), +ts.slice(11, 13));
const fmt = (ms) => {
  const d = new Date(ms);
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}T${pad(d.getUTCHours())}:00:00`;
};
const DAY = 86400000;
const WINDOW_DAYS = 3; // days shown either side of the clicked event
// <<< END ADDED

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

  // >>> ADDED: the event the user clicked (null = show the default latest-7-days chart) and its readings
  const [focus, setFocus] = useState(null); // { start, end }
  const [focusRows, setFocusRows] = useState(null); // readings around it, from the readings endpoint
  const [focusState, setFocusState] = useState("idle"); // idle | loading | ok | error
  const chartRef = useRef(null);

  useEffect(() => {
    if (!focus) {
      setFocusRows(null);
      setFocusState("idle");
      return undefined;
    }
    let cancelled = false;
    setFocusState("loading");
    dashboardApi
      .getMyReadings(token, {
        start: fmt(toMs(focus.start) - WINDOW_DAYS * DAY),
        end: fmt(toMs(focus.end) + WINDOW_DAYS * DAY + 3599000),
        limit: 2000,
      })
      .then((rows) => {
        if (cancelled) return;
        setFocusRows(rows);
        setFocusState(rows && rows.length ? "ok" : "error");
      })
      .catch(() => {
        if (!cancelled) setFocusState("error");
      });
    return () => {
      cancelled = true;
    };
  }, [focus && focus.start, focus && focus.end, token]); // eslint-disable-line react-hooks/exhaustive-deps

  // The readings around the event, with the unusual hours marked (matched by timestamp against the check's list).
  const focusReadings = useMemo(() => {
    if (!focusRows || !result) return null;
    const flagged = new Map((result.anomalies || []).map((a) => [String(a.timestamp).slice(0, 19), a]));
    return [...focusRows]
      .sort((a, b) => String(a.reading_timestamp).localeCompare(String(b.reading_timestamp)))
      .map((r) => {
        const key = String(r.reading_timestamp).slice(0, 19);
        const a = flagged.get(key);
        return {
          timestamp: key,
          consumption: r.consumption_kwh ?? 0,
          is_anomaly: !!a,
          severity: a ? a.severity : null,
          anomaly_score: a ? a.anomaly_score : null,
        };
      });
  }, [focusRows, result]);

  const pickEvent = (ev) => {
    setFocus(ev);
    setTimeout(() => chartRef.current && chartRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" }), 50);
  };
  // <<< END ADDED

  const handleCheck = async (refresh = false) => {
    setStatus("checking");
    setError(null);
    setFocus(null); // >>> ADDED: a new check starts from the default chart
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

              {/* >>> CHANGED: the chart shows the days around the clicked event when there is one, else the original
                  latest-days chart. Before: <AnomalyChart readings={result.recent_readings} /> */}
              <div ref={chartRef}>
                {focus && (
                  <p className="hint">
                    Showing {WINDOW_DAYS} days either side of {label(focus.start)}.{" "}
                    {focusState === "loading" && "Loading readings... "}
                    {focusState === "error" &&
                      "Couldn't load the readings around this event, so the latest days are shown instead. "}
                    <button className="link-button" onClick={() => setFocus(null)}>
                      Back to latest days
                    </button>
                  </p>
                )}
                <AnomalyChart
                  readings={focus && focusState === "ok" && focusReadings ? focusReadings : result.recent_readings}
                  title={focus && focusState === "ok" ? `Around ${label(focus.start)}` : undefined}
                />
              </div>
              {found > 0 && (
                <AnomalyList anomalies={result.anomalies} total={found} onSelect={pickEvent} selected={focus} />
              )}
              {/* <<< END CHANGED */}

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