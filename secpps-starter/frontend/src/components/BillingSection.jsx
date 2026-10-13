import { useCallback, useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { billingApi } from "../api/billingApi";
import StatCard from "./StatCard";
import "./BillingSection.css";

/**
 * Tariff, cost and recommendations for the logged-in user (US-20 to US-24).
 * Drop <BillingSection token={token} /> into a dashboard; it renders four panels:
 *   Electricity tariff  ->  Current cost  ->  Predicted cost  ->  Recommendations
 * The last three appear once a tariff has been saved, because nothing can be costed without one.
 */

const HORIZONS = [
  { value: "24h", label: "Next 24 hours" },
  { value: "7day", label: "Next 7 days" },
  { value: "1month", label: "Next month" },
];
const PRIORITY_LABEL = { high: "High priority", medium: "Medium priority", low: "Low priority" };

const money = (n, currency = "$") => (n == null ? "n/a" : `${currency}${Number(n).toFixed(2)}`);
const kwh = (n) => (n == null ? "n/a" : `${Number(n).toFixed(1)} kWh`);
const fmtDate = (iso) =>
  new Date(iso).toLocaleDateString("en-AU", { day: "numeric", month: "short", year: "numeric" });

/** Loads one API call; re-runs when `deps` change. Returns { data, error, loading, reload }. */
function useLoad(load, deps) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setState((s) => ({ ...s, error: null, loading: true }));
    load()
      .then((data) => {
        if (!cancelled) setState({ data, error: null, loading: false });
      })
      .catch((error) => {
        if (!cancelled) setState({ data: null, error, loading: false });
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, reloadKey]);

  const reload = useCallback(() => setReloadKey((k) => k + 1), []);
  return { ...state, reload };
}

function LoadError({ error, onRetry }) {
  return (
    <div className="status error" role="alert">
      <p>{error.message}</p>
      <button onClick={onRetry}>Try again</button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// US-20: tariff
// ---------------------------------------------------------------------------

// Same checks as the backend (app_services/tariffs.py), so mistakes are caught before sending
function validateRates(peak, offPeak) {
  for (const [name, text] of [["Peak rate", peak], ["Off-peak rate", offPeak]]) {
    const value = Number(text);
    if (String(text).trim() === "" || !Number.isFinite(value)) return `Enter the ${name.toLowerCase()}, e.g. 0.45.`;
    if (value <= 0) return `${name} must be greater than $0.`;
    if (value > 5) return `${name} looks too high. Enter it in dollars, e.g. 0.45 for 45 cents.`;
  }
  if (Number(peak) < Number(offPeak)) return "Peak rate can't be lower than the off-peak rate.";
  return null;
}

function TariffForm({ token, tariff, onSaved, onCancel }) {
  const [peak, setPeak] = useState(tariff ? String(tariff.peak_rate) : "");
  const [offPeak, setOffPeak] = useState(tariff ? String(tariff.off_peak_rate) : "");
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const problem = validateRates(peak, offPeak);
    setError(problem);
    if (problem) return;
    setSaving(true);
    try {
      await billingApi.saveTariff(token, Number(peak), Number(offPeak));
      onSaved();
    } catch (err) {
      setError(`Your tariff wasn't saved. ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form className="upload-form" onSubmit={handleSubmit} noValidate>
      <div className="field-row">
        <div className="field">
          <label htmlFor="peak-rate">Peak rate ($ per kWh)</label>
          <input
            id="peak-rate"
            type="number"
            step="0.01"
            min="0"
            placeholder="e.g. 0.52"
            value={peak}
            onChange={(e) => setPeak(e.target.value)}
          />
          <span className="hint">Charged 3:00 PM to 9:00 PM, every day</span>
        </div>
        <div className="field">
          <label htmlFor="offpeak-rate">Off-peak rate ($ per kWh)</label>
          <input
            id="offpeak-rate"
            type="number"
            step="0.01"
            min="0"
            placeholder="e.g. 0.24"
            value={offPeak}
            onChange={(e) => setOffPeak(e.target.value)}
          />
          <span className="hint">Charged at all other times</span>
        </div>
      </div>

      {error && <div className="status error" role="alert">{error}</div>}

      <div className="report-buttons">
        <button type="submit" className="billing-primary" disabled={saving}>
          {saving ? "Saving..." : "Save tariff"}
        </button>
        {onCancel && (
          <button type="button" onClick={onCancel} disabled={saving}>
            Cancel
          </button>
        )}
      </div>
    </form>
  );
}

function TariffPanel({ token, query, onSaved }) {
  const [editing, setEditing] = useState(false);
  const [justSaved, setJustSaved] = useState(false);

  const handleSaved = () => {
    setEditing(false);
    setJustSaved(true);
    onSaved();
  };

  let body;
  if (query.loading && !query.data) body = <div className="status">Loading your tariff…</div>;
  else if (query.error) body = <LoadError error={query.error} onRetry={query.reload} />;
  else if (!query.data.found)
    body = (
      <>
        <p className="hint">
          Enter the two rates from your electricity bill to see what your usage costs. You can
          change them at any time.
        </p>
        <TariffForm token={token} tariff={null} onSaved={handleSaved} />
      </>
    );
  else {
    const t = query.data.tariff;
    body = (
      <>
        {justSaved && <p className="billing-saved" role="status">Tariff saved. Your costs below use these rates.</p>}
        <div className="tariff-rates">
          <div className="tariff-rate is-peak">
            <div className="label">Peak rate</div>
            <div className="value">{money(t.peak_rate, t.currency)}/kWh</div>
            <div className="period">
              {t.peak_period.days}, {t.peak_period.start} to {t.peak_period.end}
            </div>
          </div>
          <div className="tariff-rate is-offpeak">
            <div className="label">Off-peak rate</div>
            <div className="value">{money(t.off_peak_rate, t.currency)}/kWh</div>
            <div className="period">
              {t.off_peak_period.days}, {t.off_peak_period.start} to {t.off_peak_period.end}
            </div>
          </div>
        </div>
        {editing ? (
          <TariffForm token={token} tariff={t} onSaved={handleSaved} onCancel={() => setEditing(false)} />
        ) : (
          <div className="report-buttons">
            <button
              onClick={() => {
                setEditing(true);
                setJustSaved(false);
              }}
            >
              Change rates
            </button>
          </div>
        )}
      </>
    );
  }

  return (
    <div className="chart-panel">
      <h2>Electricity tariff</h2>
      {body}
    </div>
  );
}

// ---------------------------------------------------------------------------
// US-21: current cost
// ---------------------------------------------------------------------------

/** One bar split into peak and off-peak, with the figures written underneath. */
function SplitBar({ label, peak, offPeak, format }) {
  const total = peak + offPeak;
  const peakPct = total > 0 ? (peak / total) * 100 : 0;
  return (
    <div className="split">
      <div className="split-head">
        <span>{label}</span>
        <span>{format(total)}</span>
      </div>
      <div className="split-track" role="img" aria-label={`${label}: peak ${format(peak)}, off-peak ${format(offPeak)}`}>
        {total > 0 && <div className="split-peak" style={{ width: `${peakPct}%` }} />}
        {total > 0 && <div className="split-offpeak" style={{ width: `${100 - peakPct}%` }} />}
      </div>
      <div className="split-legend">
        <span><i className="swatch is-peak" />Peak {format(peak)} ({Math.round(peakPct)}%)</span>
        <span><i className="swatch is-offpeak" />Off-peak {format(offPeak)} ({total > 0 ? 100 - Math.round(peakPct) : 0}%)</span>
      </div>
    </div>
  );
}

function CurrentCostPanel({ token, version }) {
  const query = useLoad(() => billingApi.getCurrentCost(token), [token, version]);
  const c = query.data;

  return (
    <div className="chart-panel">
      <h2>
        Current cost{" "}
        {c && (
          <span className="chart-unit">
            billing cycle {fmtDate(c.cycle_start)} to {fmtDate(c.cycle_end)}
          </span>
        )}
      </h2>
      {query.loading && !c && <div className="status">Calculating your cost…</div>}
      {query.error && <LoadError error={query.error} onRetry={query.reload} />}
      {c && (
        <>
          <div className="stat-grid">
            <StatCard label="Cost so far" value={money(c.total_cost, c.currency)} />
            <StatCard label="Peak cost" value={money(c.peak_cost, c.currency)} />
            <StatCard label="Off-peak cost" value={money(c.off_peak_cost, c.currency)} />
            <StatCard label="Peak usage" value={kwh(c.peak_kwh)} />
            <StatCard label="Off-peak usage" value={kwh(c.off_peak_kwh)} />
          </div>
          {c.peak_kwh + c.off_peak_kwh > 0 && (
            <>
              <SplitBar label="Energy used" peak={c.peak_kwh} offPeak={c.off_peak_kwh} format={kwh} />
              <SplitBar label="What it cost" peak={c.peak_cost} offPeak={c.off_peak_cost} format={(n) => money(n, c.currency)} />
            </>
          )}
          <p className="hint billing-note">{c.message}</p>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// US-22: predicted cost
// ---------------------------------------------------------------------------

function CostChart({ rows, hourly, currency }) {
  const tick = (label) =>
    hourly ? `${label.slice(11, 13)}:00` : `${label.slice(8, 10)}/${label.slice(5, 7)}`;
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="#232a35" strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="label" tickFormatter={tick} stroke="#8993a4" fontSize={12} minTickGap={16} />
        <YAxis stroke="#8993a4" fontSize={12} tickFormatter={(v) => `${currency}${v}`} />
        <Tooltip
          cursor={{ fill: "#ffffff10" }}
          contentStyle={{
            background: "#141922",
            border: "1px solid #232a35",
            borderRadius: 6,
            fontFamily: "JetBrains Mono, monospace",
            fontSize: 12,
          }}
          labelFormatter={(label) => (hourly ? new Date(label).toLocaleString() : fmtDate(label))}
          formatter={(value, name) => [money(value, currency), name]}
        />
        <Legend />
        <Bar dataKey="offpeak_cost" name="Off-peak" stackId="cost" fill="#5eead4" isAnimationActive={false} />
        <Bar dataKey="peak_cost" name="Peak" stackId="cost" fill="#f59e0b" isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function PredictedCostPanel({ query, horizon, onHorizonChange }) {
  const p = query.data;
  const noForecast = p && p.covered_hours === 0;

  return (
    <div className="chart-panel">
      <h2>
        Predicted cost{" "}
        {p && (
          <span className="chart-unit">
            {fmtDate(p.period_start)} to {fmtDate(p.period_end)}
          </span>
        )}
      </h2>

      <div className="forecast-info">
        <div className="forecast-info-row">
          <label htmlFor="cost-horizon-select">Cost horizon </label>
          <select id="cost-horizon-select" value={horizon} onChange={(e) => onHorizonChange(e.target.value)}>
            {HORIZONS.map((h) => (
              <option key={h.value} value={h.value}>
                {h.label}
              </option>
            ))}
          </select>
        </div>
        {p && (
          <div className="forecast-info-row">
            <span className="forecast-model-badge">Forecast: {p.forecast.label}</span>
          </div>
        )}
      </div>

      {query.loading && !p && <div className="status">Calculating your predicted cost…</div>}
      {query.error && <LoadError error={query.error} onRetry={query.reload} />}

      {p && noForecast && (
        <p className="hint">There is no forecast for this period yet, so there is no cost to show.</p>
      )}

      {p && !noForecast && (
        <div className={query.loading ? "billing-refreshing" : undefined}>
          <div className="stat-grid">
            <StatCard label="Predicted cost" value={money(p.predicted_total_cost, p.currency)} />
            <StatCard label="Peak cost" value={money(p.predicted_peak_cost, p.currency)} />
            <StatCard label="Off-peak cost" value={money(p.predicted_offpeak_cost, p.currency)} />
            <StatCard label="Predicted usage" value={kwh(p.predicted_total_kwh)} />
          </div>

          {p.is_partial_forecast && (
            <div className="info-note">
              <strong>Partial forecast.</strong> It covers {Math.round(p.covered_hours)} of the{" "}
              {Math.round(p.expected_hours)} hours in this period, so the real cost will be higher.
            </div>
          )}

          <CostChart rows={p.breakdown} hourly={p.breakdown_unit === "hour"} currency={p.currency} />
          <p className="hint billing-note">
            Predicted usage × your rate for each time period, per {p.breakdown_unit}.
          </p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// US-23 / US-24: recommendations
// ---------------------------------------------------------------------------

function RecommendationsPanel({ query }) {
  const a = query.data;

  return (
    <div className="chart-panel">
      <h2>Recommendations</h2>
      {query.loading && !a && <div className="status">Working out your recommendations…</div>}
      {query.error && <LoadError error={query.error} onRetry={query.reload} />}

      {a && (
        <div className={query.loading ? "billing-refreshing" : undefined}>
          {a.estimated_total_saving > 0 && (
            <div className="stat-grid">
              <StatCard label="Predicted cost" value={money(a.predicted_cost.predicted_total_cost)} />
              <StatCard label="If you follow the advice" value={money(a.recommended_cost)} />
              <StatCard label="Possible saving" value={money(a.estimated_total_saving)} />
            </div>
          )}

          {a.recommendations.length === 0 ? (
            <p className="hint">
              Nothing to recommend right now: your predicted usage for this period doesn't call for
              any changes.
            </p>
          ) : (
            <ul className="rec-list">
              {a.recommendations.map((r) => (
                <li key={r.rule} className={`rec-card is-${r.priority}`}>
                  <div className="rec-head">
                    <strong>{r.title}</strong>
                    <span className={`rec-priority is-${r.priority}`}>
                      {PRIORITY_LABEL[r.priority] || r.priority}
                    </span>
                  </div>
                  <p className="rec-message">{r.message}</p>
                  <dl className="rec-facts">
                    <div>
                      <dt>Why</dt>
                      <dd>{r.explanation}</dd>
                    </div>
                    <div>
                      <dt>Expected benefit</dt>
                      <dd>{r.expected_benefit}</dd>
                    </div>
                  </dl>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

function Outlook({ token, version }) {
  const [horizon, setHorizon] = useState("7day");
  const predicted = useLoad(() => billingApi.getPredictedCost(token, horizon), [token, horizon, version]);
  const advice = useLoad(() => billingApi.getAdvice(token, horizon), [token, horizon, version]);
  return (
    <>
      <PredictedCostPanel query={predicted} horizon={horizon} onHorizonChange={setHorizon} />
      <RecommendationsPanel query={advice} />
    </>
  );
}

export default function BillingSection({ token }) {
  const tariff = useLoad(() => billingApi.getTariff(token), [token]);
  const [version, setVersion] = useState(0); // bumped after a save so the cost panels recalculate

  const handleSaved = () => {
    tariff.reload();
    setVersion((v) => v + 1);
  };

  return (
    <>
      <TariffPanel token={token} query={tariff} onSaved={handleSaved} />
      {tariff.data?.found && (
        <>
          <CurrentCostPanel token={token} version={version} />
          <Outlook token={token} version={version} />
        </>
      )}
    </>
  );
}