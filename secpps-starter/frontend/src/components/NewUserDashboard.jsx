import { useMemo, useState } from "react";
import StatCard from "./StatCard";
import ForecastInfo from "./ForecastInfo";
import NewUserForecastChart from "./NewUserForecastChart";
// >>> ADDED: live meter demo
import { useLiveMeter, LiveControls, recalcHourly, toDaily } from "./LiveDemo";
// <<< END ADDED
import AnomalyPlaceholder from "./AnomalyPlaceholder";
import ReportButtons from "./ReportButtons";
import Warnings from "./Warnings";

const fmtDate = (iso) =>
  new Date(iso).toLocaleDateString("en-AU", { day: "numeric", month: "short", year: "numeric" });

const kwh = (value, digits) => (value == null ? "n/a" : `${Number(value).toFixed(digits)} kWh`);

/** Dashboard for a finished upload: history + forecast, horizon, model used, notes, reports. */
export default function NewUserDashboard({ data, token }) {
  // >>> CHANGED: `history` is renamed `baseHistory`; `history` below now = uploaded history + demo readings.
  // (Before: const { history, result, info, warnings } = data;)
  const { history: baseHistory, result, info, warnings } = data;
  // <<< END CHANGED
  const [horizon, setHorizon] = useState("24h");

  // >>> ADDED: live meter demo. Extra readings (one per 30 s) are appended to the uploaded history, so the graph,
  // the stat cards and the forecasts all update. The saved result from the server is left untouched.
  const meter = useLiveMeter(useMemo(() => baseHistory.map((p) => ({ ts: p.timestamp, kwh: p.kwh })), [baseHistory]));
  const history = useMemo(
    () => (meter.live.length ? [...baseHistory, ...meter.live.map((p) => ({ timestamp: p.ts, kwh: p.kwh }))] : baseHistory),
    [baseHistory, meter.live]
  );
  // <<< END ADDED

  const stats = useMemo(() => {
    let total = 0;
    let peak = history[0];
    for (const p of history) {
      total += p.kwh;
      if (p.kwh > peak.kwh) peak = p;
    }
    const days = history.length / 24;
    return { days, avgDaily: total / days, peak };
  }, [history]);

  // >>> CHANGED: once live readings exist, each horizon is recalculated from them (24h hourly, 7day / 1month daily).
  // Before: const horizons = result.horizons || {};
  const savedHorizons = result.horizons || {};
  const horizons = useMemo(() => {
    if (!meter.live.length) return savedHorizons;
    const out = { ...savedHorizons };
    const redo = (key, make) => {
      const saved = savedHorizons[key];
      if (!saved || saved.error) return;
      const predictions = make();
      if (!predictions.length) return;
      out[key] = { ...saved, predictions, total_kwh: predictions.reduce((s, p) => s + p.predicted_kwh, 0) };
    };
    redo("24h", () => recalcHourly(meter.series, 24).map((p) => ({ timestamp: p.ts, predicted_kwh: p.kwh })));
    redo("7day", () => toDaily(recalcHourly(meter.series, 7 * 24 + 24), 7));
    redo("1month", () => toDaily(recalcHourly(meter.series, 30 * 24 + 24), 30));
    return out;
  }, [savedHorizons, meter.live.length, meter.series]);
  // <<< END CHANGED
  const current = horizons[horizon];
  const placeName = (result.location && result.location.resolved_name) || data.location;
  const modelText = current && !current.error ? `${current.method} (fine-tuned on your data)` : null;

  return (
    <>
      <div className="household-info">
        Forecast for <strong>{placeName}</strong> · built from {Math.round(stats.days)} days of your
        data ({fmtDate(info.start)} to {fmtDate(info.end)})
      </div>

      <Warnings items={warnings} />

      <div className="stat-grid">
        <StatCard label="Average per day" value={kwh(stats.avgDaily, 1)} />
        <StatCard label="Peak hour" value={kwh(stats.peak.kwh, 2)} />
        <StatCard label="Next 24 hours" value={kwh(horizons["24h"]?.total_kwh, 1)} />
        <StatCard label="Next 7 days" value={kwh(horizons["7day"]?.total_kwh, 1)} />
        <StatCard label="Next 30 days" value={kwh(horizons["1month"]?.total_kwh, 1)} />
      </div>

      <ForecastInfo horizon={horizon} onHorizonChange={setHorizon} model={modelText} />
      {/* >>> ADDED: live meter controls (sit right above the graph) */}
      <LiveControls meter={meter} />
      {/* <<< END ADDED */}
      <NewUserForecastChart history={history} horizon={horizon} horizonData={current} />

      {result.limitations && result.limitations.length > 0 && (
        <div className="chart-panel">
          <h2>About this forecast</h2>
          <ul className="limitations">
            {result.limitations.map((text) => (
              <li key={text}>{text}</li>
            ))}
          </ul>
        </div>
      )}

      <AnomalyPlaceholder />
      <ReportButtons token={token} source="upload" />
    </>
  );
}