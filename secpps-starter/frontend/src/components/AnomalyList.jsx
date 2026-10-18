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


const HOUR = 3600000;
const toMs = (ts) => Date.UTC(+ts.slice(0, 4), +ts.slice(5, 7) - 1, +ts.slice(8, 10), +ts.slice(11, 13));

function groupEvents(anomalies) {
  const asc = [...anomalies].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  const events = [];
  asc.forEach((a) => {
    const last = events[events.length - 1];
    if (last && toMs(a.timestamp) - toMs(last.end) <= HOUR) {
      last.end = a.timestamp;
      last.hours += 1;
      if (Number(a.consumption) > last.peak) last.peak = Number(a.consumption);
      if ((ORDER[a.severity] ?? 3) < (ORDER[last.severity] ?? 3)) last.severity = a.severity;
      if (Number(a.anomaly_score) > last.score) { last.score = Number(a.anomaly_score); last.evidence = a.model_evidence; }
    } else {
      events.push({
        start: a.timestamp, end: a.timestamp, hours: 1, peak: Number(a.consumption),
        severity: a.severity, score: Number(a.anomaly_score), evidence: a.model_evidence,
      });
    }
  });
  return events.sort(
    (a, b) => (ORDER[a.severity] ?? 3) - (ORDER[b.severity] ?? 3) || b.start.localeCompare(a.start)
  );
}

const range = (e) =>
  e.start === e.end
    ? label(e.start)
    : e.start.slice(0, 10) === e.end.slice(0, 10)
      ? `${label(e.start)} to ${e.end.slice(11, 16)}`
      : `${label(e.start)} to ${label(e.end)}`;



export default function AnomalyList({ anomalies, total, message, onSelect, selected }) {

  const [showAll, setShowAll] = useState(false);
  
  const [grouped, setGrouped] = useState(true);
 

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
  
  const events = groupEvents(anomalies);
  const items = grouped ? events : sorted;
  const visible = showAll ? items : items.slice(0, SHOW_FIRST);
  const isSelected = (start, end) => selected && selected.start === start && selected.end === end;
  const clickable = (start, end) =>
    onSelect
      ? {
          onClick: () => onSelect({ start, end }),
          onKeyDown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect({ start, end }); } },
          role: "button",
          tabIndex: 0,
          title: "Show this on the chart",
          style: {
            cursor: "pointer",
            ...(isSelected(start, end) ? { outline: "1px solid #5eead4" } : {}),
          },
        }
      : {};
 
  const found = total ?? anomalies.length;

  return (
    <div className="anomaly-list-block">
      <h3>Unusual readings ({found})</h3>
      {found > anomalies.length && (
        <p className="hint">Showing the most recent {anomalies.length} of {found}.</p>
      )}
      {/* >>> grouping switch + click hint */}
      <p className="hint">
        {grouped
          ? `Grouped into ${events.length} ${events.length === 1 ? "event" : "events"} (consecutive hours count as one). `
          : "One row per hour. "}
        <button className="link-button" onClick={() => { setGrouped((v) => !v); setShowAll(false); }}>
          {grouped ? "Show each hour" : "Group by event"}
        </button>
        {onSelect && " Click a row to see it on the chart."}
      </p>
      {/* <<< END */}
      <div className="anomaly-list">
     
          grouped ? (
            <div key={a.start} className="anomaly-row" {...clickable(a.start, a.end)}>
              <span className="anomaly-severity" style={{ background: SEVERITY_COLORS[a.severity] || "#64748b" }}>
                {a.severity}
              </span>
              <span className="anomaly-timestamp">{range(a)}</span>
              <span className="anomaly-consumption">
                {a.hours} {a.hours === 1 ? "hour" : "hours"}, peak {a.peak.toFixed(3)} kWh
              </span>
              <span className="anomaly-score">score {a.score.toFixed(2)}</span>
              <span className="anomaly-evidence">{EVIDENCE[a.evidence] || ""}</span>
            </div>
          ) : (
            <div key={a.timestamp} className="anomaly-row" {...clickable(a.timestamp, a.timestamp)}>
              <span className="anomaly-severity" style={{ background: SEVERITY_COLORS[a.severity] || "#64748b" }}>
                {a.severity}
              </span>
              <span className="anomaly-timestamp">{label(a.timestamp)}</span>
              <span className="anomaly-consumption">{Number(a.consumption).toFixed(3)} kWh</span>
              <span className="anomaly-score">score {Number(a.anomaly_score).toFixed(2)}</span>
              <span className="anomaly-evidence">{EVIDENCE[a.model_evidence] || ""}</span>
            </div>
          )
        
       
      </div>
      {items.length > SHOW_FIRST && (
        <button className="link-button anomaly-more" onClick={() => setShowAll((v) => !v)}>
          {showAll ? "Show fewer" : `Show all ${items.length}`}
        </button>
      )}
    </div>
  );
}