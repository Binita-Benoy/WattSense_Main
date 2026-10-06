import { useEffect, useState } from "react";

// Same wording the backend reports in `progress`, in the order the model runs
const STEPS = [
  "Finding your location",
  "Fetching past weather for your dates",
  "Fetching the weather forecast",
  "Preparing your data",
  "Training on your data and forecasting 24 hours + 7 days",
  "Training on your data and forecasting 1 month",
];

function useElapsed(startIso) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);
  if (!startIso) return null;
  const seconds = Math.max(0, Math.floor((now - new Date(startIso).getTime()) / 1000));
  return `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, "0")}s`;
}

export default function ForecastProgress({ job }) {
  const elapsed = useElapsed(job.started_at);
  const current = STEPS.indexOf(job.progress);
  const queued = job.status === "queued";

  return (
    <div className="chart-panel">
      <h2>Building your forecast</h2>

      {queued ? (
        <p className="hint">
          Waiting for your turn. Only one forecast is built at a time, so yours starts as soon as
          the current one finishes.
        </p>
      ) : (
        <p className="hint">
          This usually takes 1 to 4 minutes{elapsed ? ` (running for ${elapsed})` : ""}. You can
          leave this page open. If you close it, your forecast keeps building and will be here when
          you come back.
        </p>
      )}

      <div className="progress-now" role="status">
        <span className="spinner" aria-hidden="true" />
        <span>{queued ? "In the queue" : job.progress || "Starting"}</span>
      </div>

      {!queued && (
        <ol className="progress-steps">
          {STEPS.map((label, i) => (
            <li key={label} className={i < current ? "done" : i === current ? "current" : ""}>
              {label}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}