import { useMemo, useState } from "react";
import StatCard from "./StatCard";
import ForecastInfo from "./ForecastInfo";
import NewUserForecastChart from "./NewUserForecastChart";
import { useLiveMeter, LiveControls, recalcHourly, toDaily } from "./LiveDemo";
import AnomalyCheck from "./AnomalyCheck"; 
import ReportButtons from "./ReportButtons";
import Warnings from "./Warnings";

const fmtDate = (iso) =>
  new Date(iso).toLocaleDateString("en-AU", { day: "numeric", month: "short", year: "numeric" });

const kwh = (value, digits) => (value == null ? "n/a" : `${Number(value).toFixed(digits)} kWh`);

/** Dashboard for a finished upload: history + forecast, horizon, model used, notes, reports. */
export default function NewUserDashboard({ data, token }) {
 
  const { history: baseHistory, result, info, warnings } = data;
 
  const [horizon, setHorizon] = useState("24h");

  
  const meter = useLiveMeter(useMemo(() => baseHistory.map((p) => ({ ts: p.timestamp, kwh: p.kwh })), [baseHistory]));
  const history = useMemo(
    () => (meter.live.length ? [...baseHistory, ...meter.live.map((p) => ({ timestamp: p.ts, kwh: p.kwh }))] : baseHistory),
    [baseHistory, meter.live]
  );
 

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

      
      <ForecastInfo horizon={horizon} onHorizonChange={setHorizon} model={null} />
      
      <LiveControls meter={meter} />
     
      <NewUserForecastChart history={history} horizon={horizon} horizonData={current} />

     

      <AnomalyCheck token={token} />
      <ReportButtons token={token} source="upload" />
    </>
  );
}