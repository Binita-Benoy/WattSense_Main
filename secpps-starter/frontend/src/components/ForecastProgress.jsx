import { useEffect, useState } from "react";

// Same wording the backend reports in `progress`, in the order the model runs
// >>> CHANGED: only 4 steps are shown. The two "Training on your data..." stages are still reported by the
// backend, but they are displayed as part of "Preparing your data". Original list was:
//   "Finding your location",
//   "Fetching past weather for your dates",
//   "Fetching the weather forecast",
//   "Preparing your data",
//   "Training on your data and forecasting 24 hours + 7 days",
//   "Training on your data and forecasting 1 month",
const STEPS = [
  "Finding your location",
  "Fetching past weather for your dates",
  "Fetching the weather forecast",
  "Preparing your data",
];

// Backend wording that should be shown as a different (visible) step
const SHOWN_AS = {
  "Training on your data and forecasting 24 hours + 7 days": "Preparing your data",
  "Training on your data and forecasting 1 month": "Preparing your data",
};
// <<< END CHANGED

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
  // >>> CHANGED: translate the backend wording to a visible step (was: STEPS.indexOf(job.progress))
  const shownProgress = SHOWN_AS[job.progress] || job.progress;
  const current = STEPS.indexOf(shownProgress);
  // <<< END CHANGED
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
        {/* >>> CHANGED: show the translated text (was: job.progress || "Starting") */}
        <span>{queued ? "In the queue" : shownProgress || "Starting"}</span>
        {/* <<< END CHANGED */}
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