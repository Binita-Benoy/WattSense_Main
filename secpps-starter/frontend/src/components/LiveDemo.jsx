// >>> ADDED: live-meter DEMO (whole file is new). Browser only: no backend, no database, no model.
// It does NOT draw its own chart. The dashboards feed the extra readings into their EXISTING graphs, so the
// usage line grows and the forecast line is recalculated on the same graphs every 30 s.
import { useEffect, useMemo, useRef, useState } from "react";

export const TICK_MS = 30000; // one new reading every 30 s (demo); each one stands for the next hour

// Timestamps are handled as plain text ("2013-07-31T23:00:00"), never converted to UTC, so they line up
// exactly with the timestamps the API already returns.
const pad = (n) => String(n).padStart(2, "0");
const parse = (ts) => Date.UTC(+ts.slice(0, 4), +ts.slice(5, 7) - 1, +ts.slice(8, 10), +ts.slice(11, 13));
const fmt = (ms) => {
  const d = new Date(ms);
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}T${pad(d.getUTCHours())}:00:00`;
};
const hourOf = (ts) => +ts.slice(11, 13);
const avg = (a) => a.reduce((s, v) => s + v, 0) / a.length;

/**
 * Hourly forecast for the next `n` hours: the typical value for each hour of day (last 7 days, live readings
 * included), scaled by how the latest 24 h compare with the 24 h before. A new reading therefore moves it.
 * series = [{ ts, kwh }]  ->  [{ ts, kwh }]
 */
export function recalcHourly(series, n) {
  if (series.length < 48) return [];
  const last = series[series.length - 1];
  const byHour = {};
  series.slice(-24 * 7).forEach((p) => {
    (byHour[hourOf(p.ts)] = byHour[hourOf(p.ts)] || []).push(p.kwh);
  });
  const now = avg(series.slice(-24).map((p) => p.kwh));
  const before = avg(series.slice(-48, -24).map((p) => p.kwh));
  const scale = before > 0 ? Math.min(1.5, Math.max(0.67, now / before)) : 1;
  const out = [];
  for (let i = 1; i <= n; i++) {
    const ts = fmt(parse(last.ts) + i * 3600000);
    out.push({ ts, kwh: avg(byHour[hourOf(ts)] || [now]) * scale });
  }
  return out;
}

/** Hourly -> daily totals ({date, predicted_kwh}), starting at the next midnight (for the 7-day / 1-month views). */
export function toDaily(hourly, days) {
  const start = hourly.findIndex((p) => hourOf(p.ts) === 0);
  if (start < 0) return [];
  const out = [];
  for (let d = 0; d < days; d++) {
    const chunk = hourly.slice(start + d * 24, start + d * 24 + 24);
    if (chunk.length < 24) break;
    out.push({ date: chunk[0].ts.slice(0, 10), predicted_kwh: chunk.reduce((s, p) => s + p.kwh, 0) });
  }
  return out;
}

/** Saved horizons -> horizons recalculated from the series once live readings exist (24h hourly, 7day / 1month daily). */
export function liveHorizons(saved, series, hasLive) {
  if (!hasLive) return saved;
  const out = { ...saved };
  const redo = (key, make) => {
    const old = saved[key];
    if (!old || old.error) return;
    const predictions = make();
    if (!predictions.length) return;
    out[key] = { ...old, predictions, total_kwh: predictions.reduce((s, p) => s + p.predicted_kwh, 0) };
  };
  redo("24h", () => recalcHourly(series, 24).map((p) => ({ timestamp: p.ts, predicted_kwh: p.kwh })));
  redo("7day", () => toDaily(recalcHourly(series, 7 * 24 + 24), 7));
  redo("1month", () => toDaily(recalcHourly(series, 30 * 24 + 24), 30));
  return out;
}

/**
 * The live meter. `base` = [{ ts, kwh }] the dashboard already has.
 * Returns the extra readings plus the controls' state. Changing `base` (new date range) clears the demo.
 */
export function useLiveMeter(base) {
  const [live, setLive] = useState([]);
  const [running, setRunning] = useState(false);
  const [left, setLeft] = useState(TICK_MS / 1000);

  const series = useMemo(() => [...base, ...live], [base, live]);
  const seriesRef = useRef(series);
  seriesRef.current = series;

  // No spikes: this household's average for that hour over the last 7 days, +-5%, kept inside its own min..max.
  const addNow = () => {
    const s = seriesRef.current;
    if (s.length < 48) return;
    const ts = fmt(parse(s[s.length - 1].ts) + 3600000);
    const same = s.slice(-24 * 7).filter((p) => hourOf(p.ts) === hourOf(ts)).map((p) => p.kwh);
    if (!same.length) return;
    const kwh = Math.min(Math.max(avg(same) * (0.95 + Math.random() * 0.1), Math.min(...same)), Math.max(...same));
    setLive((l) => [...l, { ts, kwh }]);
    setLeft(TICK_MS / 1000);
  };

  useEffect(() => {
    if (!running) return undefined;
    const add = setInterval(addNow, TICK_MS);
    const count = setInterval(() => setLeft((n) => (n > 1 ? n - 1 : TICK_MS / 1000)), 1000);
    return () => { clearInterval(add); clearInterval(count); };
  }, [running]); // eslint-disable-line react-hooks/exhaustive-deps

  const baseKey = `${base.length}|${base.length ? base[base.length - 1].ts : ""}`;
  useEffect(() => { setLive([]); setRunning(false); setLeft(TICK_MS / 1000); }, [baseKey]);

  const reset = () => { setLive([]); setRunning(false); setLeft(TICK_MS / 1000); };
  return { live, series, running, left, toggle: () => setRunning((r) => !r), addNow, reset };
}

/** Small control bar shown above the graph. */
export function LiveControls({ meter }) {
  const { live, running, left, toggle, addNow, reset } = meter;
  return (
    <div className="household-info" style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
      <strong>Live meter (demo)</strong>
      <button onClick={toggle}>{running ? "Pause" : live.length ? "Resume" : "Start live feed"}</button>
      <button onClick={addNow}>Add reading now</button>
      {live.length > 0 && <button onClick={reset}>Reset</button>}
      <span style={{ color: "#8993a4", fontSize: 13 }}>
        {running ? `next reading in ${left}s · ` : ""}{live.length} live reading{live.length === 1 ? "" : "s"} added ·
        forecast recalculated on each one (demo, not the trained model)
      </span>
    </div>
  );
}
// <<< END ADDED