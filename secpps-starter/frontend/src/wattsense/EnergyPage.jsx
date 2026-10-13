import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { billingApi } from "../api/billingApi";
import { useAuth } from "../context/AuthContext";
import { Link, clockToHour, day, hourLabel, kwh, money, shortDay, useApi } from "./lib";

/**
 * Energy page (US-20 to US-24): tariff, current cost, predicted cost and recommendations.
 *
 *   hero          tariff clock + what this cycle has cost + what's coming
 *   tariff        the two rates and when each applies; change them here
 *   this cycle    where the money went, peak vs off-peak
 *   forecast      predicted cost for the chosen period, per day (or hour)
 *   advice        recommendations, each with why, expected benefit and priority
 */

const PEAK = "#F59E0B";
const OFFPEAK = "#5EEAD4";
const DEFAULT_WINDOW = [15, 21]; // standard peak window, shown before a tariff exists

const HORIZONS = [
  { value: "24h", label: "24 hours" },
  { value: "7day", label: "7 days" },
  { value: "1month", label: "30 days" },
];
const horizonText = { "24h": "next 24 hours", "7day": "next 7 days", "1month": "next 30 days" };
const PRIORITY = { high: "High priority", medium: "Medium priority", low: "Low priority" };

const isNoTariff = (error) => error?.status === 404 && /tariff/i.test(error.message);

// ---------------------------------------------------------------------------
// Tariff clock: the day as a 24-hour dial, peak hours in amber, a marker at "now".
// ---------------------------------------------------------------------------
const SIZE = 280;
const C = SIZE / 2;
const R = 112;

function polar(r, hour) {
  const a = ((hour / 24) * 360 - 90) * (Math.PI / 180);
  return [C + r * Math.cos(a), C + r * Math.sin(a)];
}

function arcPath(h0, h1, r) {
  const [x0, y0] = polar(r, h0);
  const [x1, y1] = polar(r, h1);
  const large = h1 - h0 > 12 ? 1 : 0;
  return `M ${x0} ${y0} A ${r} ${r} 0 ${large} 1 ${x1} ${y1}`;
}

function useNowHour() {
  const read = () => {
    const d = new Date();
    return d.getHours() + d.getMinutes() / 60;
  };
  const [now, setNow] = useState(read);
  useEffect(() => {
    const id = setInterval(() => setNow(read()), 60_000);
    return () => clearInterval(id);
  }, []);
  return now;
}

function TariffClock({ tariff }) {
  const now = useNowHour();
  const start = tariff ? clockToHour(tariff.peak_period.start) : DEFAULT_WINDOW[0];
  const end = tariff ? clockToHour(tariff.peak_period.end) : DEFAULT_WINDOW[1];
  const inPeak = now >= start && now < end;
  const [hx0, hy0] = polar(R - 30, now); // the "now" needle sits across the ring, clear of the centre text
  const [hx, hy] = polar(R + 12, now);
  const [dx, dy] = polar(R, now);

  const ticks = Array.from({ length: 24 }, (_, h) => {
    const major = h % 6 === 0;
    const [x0, y0] = polar(R + 14, h);
    const [x1, y1] = polar(R + (major ? 22 : 18), h);
    return <line key={h} x1={x0} y1={y0} x2={x1} y2={y1} className={major ? "clock-tick is-major" : "clock-tick"} />;
  });
  const labels = [0, 6, 12, 18].map((h) => {
    const [x, y] = polar(R + 36, h);
    return (
      <text key={h} x={x} y={y} className="clock-hour" textAnchor="middle" dominantBaseline="middle">
        {hourLabel(h)}
      </text>
    );
  });

  const rate = tariff ? (inPeak ? tariff.peak_rate : tariff.off_peak_rate) : null;
  const summary = `Peak rate from ${hourLabel(start)} to ${hourLabel(end)}, off-peak at all other times. It is now ${inPeak ? "peak" : "off-peak"} time.`;

  return (
    <figure className="clock" aria-label={summary}>
      <svg viewBox={`-40 -40 ${SIZE + 80} ${SIZE + 80}`} role="img" aria-hidden="true">
        <circle cx={C} cy={C} r={R} className="clock-track" />
        <path d={arcPath(end, start + 24, R)} className="clock-arc is-offpeak" pathLength="1" />
        <path d={arcPath(start, end, R)} className="clock-arc is-peak" pathLength="1" />
        {ticks}
        {labels}
        <line x1={hx0} y1={hy0} x2={hx} y2={hy} className="clock-hand" />
        <circle cx={dx} cy={dy} r="8" className={`clock-now ${inPeak ? "is-peak" : "is-offpeak"}`} />
      </svg>
      <figcaption className="clock-face">
        <span className="clock-face-label">Right now</span>
        <strong className={inPeak ? "is-peak" : "is-offpeak"}>{inPeak ? "Peak" : "Off-peak"}</strong>
        {rate != null && <span className="clock-face-rate">{money(rate, tariff.currency)} per kWh</span>}
      </figcaption>
    </figure>
  );
}

// ---------------------------------------------------------------------------
// Rates form (US-20)
// ---------------------------------------------------------------------------
function validate(peak, offPeak) {
  const errors = {};
  const check = (key, text) => {
    if (String(text).trim() === "") errors[key] = "Enter a rate.";
    else if (!Number.isFinite(Number(text))) errors[key] = "Enter a number, for example 0.45.";
    else if (Number(text) <= 0) errors[key] = "The rate must be more than $0.";
    else if (Number(text) > 5) errors[key] = "Enter dollars, for example 0.45 for 45 cents.";
  };
  check("peak", peak);
  check("offPeak", offPeak);
  if (!errors.peak && !errors.offPeak && Number(peak) < Number(offPeak))
    errors.peak = "The peak rate can't be lower than the off-peak rate.";
  return errors;
}

function RateInput({ id, label, hint, tone, value, onChange, error }) {
  return (
    <div className="rate-input">
      <label htmlFor={id}>
        <i className={`swatch is-${tone}`} aria-hidden="true" />
        {label}
      </label>
      <div className={`rate-input-box ${error ? "has-error" : ""}`}>
        <span aria-hidden="true">$</span>
        <input
          id={id}
          inputMode="decimal"
          autoComplete="off"
          placeholder="0.00"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          aria-invalid={!!error}
          aria-describedby={`${id}-help`}
        />
        <span aria-hidden="true">per kWh</span>
      </div>
      <p id={`${id}-help`} className={error ? "field-msg is-error" : "field-msg"}>
        {error || hint}
      </p>
    </div>
  );
}

function RatesForm({ tariff, onSaved, onCancel }) {
  const { token } = useAuth();
  const [peak, setPeak] = useState(tariff ? String(tariff.peak_rate) : "");
  const [offPeak, setOffPeak] = useState(tariff ? String(tariff.off_peak_rate) : "");
  const [errors, setErrors] = useState({});
  const [saving, setSaving] = useState(false);
  const [failed, setFailed] = useState(null);

  async function submit(e) {
    e.preventDefault();
    const found = validate(peak, offPeak);
    setErrors(found);
    setFailed(null);
    if (Object.keys(found).length) return;
    setSaving(true);
    try {
      await billingApi.saveTariff(token, Number(peak), Number(offPeak));
      onSaved();
    } catch (err) {
      setFailed(err.message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <form className="rates-form" onSubmit={submit} noValidate>
      <div className="rates-form-fields">
        <RateInput id="peak-rate" label="Peak rate" tone="peak" hint="3pm to 9pm, every day" value={peak} onChange={setPeak} error={errors.peak} />
        <RateInput id="offpeak-rate" label="Off-peak rate" tone="offpeak" hint="All other hours" value={offPeak} onChange={setOffPeak} error={errors.offPeak} />
      </div>
      {failed && (
        <p className="field-msg is-error" role="alert">
          The rates weren't saved: {failed}
        </p>
      )}
      <div className="ws-actions">
        <button className="ws-btn" type="submit" disabled={saving}>
          {saving ? "Saving…" : "Save rates"}
        </button>
        {onCancel && (
          <button className="ws-btn ws-btn-quiet" type="button" onClick={onCancel} disabled={saving}>
            Cancel
          </button>
        )}
      </div>
    </form>
  );
}

// ---------------------------------------------------------------------------
// Small shared pieces
// ---------------------------------------------------------------------------
function Problem({ error, onRetry }) {
  if (error?.status === 404)
    return (
      <div className="problem">
        <p>{error.message}</p>
        <Link to="/" className="ws-btn ws-btn-quiet">
          Go to the dashboard
        </Link>
      </div>
    );
  return (
    <div className="problem is-error" role="alert">
      <p>{error.message}</p>
      <button className="ws-btn ws-btn-quiet" onClick={onRetry}>
        Try again
      </button>
    </div>
  );
}

function Split({ label, peak, offPeak, format }) {
  const total = peak + offPeak;
  const pct = total > 0 ? (peak / total) * 100 : 0;
  return (
    <div className="split">
      <div className="split-head">
        <span>{label}</span>
        <strong>{format(total)}</strong>
      </div>
      <div className="split-bar" role="img" aria-label={`${label}: peak ${format(peak)}, off-peak ${format(offPeak)}`}>
        {total > 0 && <span className="is-peak" style={{ width: `${pct}%` }} />}
        {total > 0 && <span className="is-offpeak" style={{ width: `${100 - pct}%` }} />}
      </div>
      <div className="split-legend">
        <span>
          <i className="swatch is-peak" /> Peak {format(peak)} <em>{Math.round(pct)}%</em>
        </span>
        <span>
          <i className="swatch is-offpeak" /> Off-peak {format(offPeak)} <em>{total > 0 ? 100 - Math.round(pct) : 0}%</em>
        </span>
      </div>
    </div>
  );
}

function HorizonSwitch({ value, onChange }) {
  return (
    <div className="horizon" role="group" aria-label="Forecast period">
      {HORIZONS.map((h) => (
        <button key={h.value} className={h.value === value ? "is-active" : ""} aria-pressed={h.value === value} onClick={() => onChange(h.value)}>
          {h.label}
        </button>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sections
// ---------------------------------------------------------------------------
function Hero({ tariff, cost, predicted, advice, horizon }) {
  const c = cost.data;
  const p = predicted.data;
  const a = advice.data;

  return (
    <section className="hero" aria-labelledby="hero-title">
      <TariffClock tariff={tariff} />
      <div className="hero-copy">
        <h2 id="hero-title" className="hero-kicker">
          {c ? `This billing cycle, ${shortDay(c.cycle_start)} to ${shortDay(c.cycle_end)}` : "This billing cycle"}
        </h2>
        {cost.error ? (
          <Problem error={cost.error} onRetry={cost.reload} />
        ) : (
          <>
            <p className="hero-figure">{c ? money(c.total_cost, c.currency) : "…"}</p>
            {c && (
              <p className="hero-note">
                {c.is_historical ? c.message : "Running total. It grows as new readings arrive."}
              </p>
            )}
            <dl className="hero-facts">
              <div>
                <dt>Predicted, {horizonText[horizon]}</dt>
                <dd>{p && p.covered_hours > 0 ? money(p.predicted_total_cost, p.currency) : "—"}</dd>
              </div>
              <div>
                <dt>You could save</dt>
                <dd className="is-offpeak">{a && a.estimated_total_saving > 0 ? money(a.estimated_total_saving) : "—"}</dd>
              </div>
            </dl>
          </>
        )}
      </div>
    </section>
  );
}

function SetupHero({ onSaved }) {
  return (
    <section className="hero is-setup" aria-labelledby="setup-title">
      <TariffClock tariff={null} />
      <div className="hero-copy">
        <h2 id="setup-title" className="hero-title">Add your electricity rates</h2>
        <p className="hero-note">
          Your bill charges one rate from 3pm to 9pm and a cheaper one the rest of the day. Enter both and we'll
          work out what your usage costs, what it's likely to cost next, and where you can save.
        </p>
        <RatesForm tariff={null} onSaved={onSaved} />
      </div>
    </section>
  );
}

function TariffSection({ tariff, onSaved }) {
  const [editing, setEditing] = useState(false);
  const [saved, setSaved] = useState(false);
  const t = tariff;
  return (
    <section className="panel" aria-labelledby="tariff-title">
      <div className="panel-head">
        <h2 id="tariff-title">Your tariff</h2>
        {!editing && (
          <button
            className="ws-btn ws-btn-quiet"
            onClick={() => {
              setEditing(true);
              setSaved(false);
            }}
          >
            Change rates
          </button>
        )}
      </div>
      {saved && (
        <p className="saved-note" role="status">
          Rates saved. Every figure on this page now uses them.
        </p>
      )}
      {editing ? (
        <RatesForm
          tariff={t}
          onCancel={() => setEditing(false)}
          onSaved={() => {
            setEditing(false);
            setSaved(true);
            onSaved();
          }}
        />
      ) : (
        <ul className="rates">
          <li>
            <span className="rates-name">
              <i className="swatch is-peak" /> Peak
            </span>
            <span className="rates-when">
              {t.peak_period.start} to {t.peak_period.end}, every day
            </span>
            <strong>{money(t.peak_rate, t.currency)}</strong>
            <span className="rates-unit">per kWh</span>
          </li>
          <li>
            <span className="rates-name">
              <i className="swatch is-offpeak" /> Off-peak
            </span>
            <span className="rates-when">All other hours</span>
            <strong>{money(t.off_peak_rate, t.currency)}</strong>
            <span className="rates-unit">per kWh</span>
          </li>
        </ul>
      )}
      {!editing && t.peak_rate > t.off_peak_rate && (
        <p className="panel-note">
          Each kWh moved out of peak hours saves {money(t.peak_rate - t.off_peak_rate, t.currency)}. Peak costs{" "}
          {(t.peak_rate / t.off_peak_rate).toFixed(1)} times as much.
        </p>
      )}
    </section>
  );
}

function CycleSection({ cost }) {
  const c = cost.data;
  return (
    <section className="panel" aria-labelledby="cycle-title">
      <div className="panel-head">
        <h2 id="cycle-title">Where this cycle's money went</h2>
      </div>
      {cost.error && <Problem error={cost.error} onRetry={cost.reload} />}
      {!c && !cost.error && <p className="panel-note">Adding up your readings…</p>}
      {c && c.peak_kwh + c.off_peak_kwh === 0 && <p className="panel-note">{c.message}</p>}
      {c && c.peak_kwh + c.off_peak_kwh > 0 && (
        <>
          <Split label="Energy used" peak={c.peak_kwh} offPeak={c.off_peak_kwh} format={kwh} />
          <Split label="What it cost" peak={c.peak_cost} offPeak={c.off_peak_cost} format={(n) => money(n, c.currency)} />
          <p className="panel-note">
            Peak hours take a bigger share of the cost than of the energy, because each peak kWh is charged at the
            higher rate.
          </p>
        </>
      )}
    </section>
  );
}

function CostTooltip({ active, payload, hourly }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tip">
      <strong>{hourly ? new Date(d.label).toLocaleString("en-AU", { weekday: "short", hour: "numeric" }) : day(d.label)}</strong>
      <span>
        <i className="swatch is-peak" /> Peak {money(d.peak_cost)}
      </span>
      <span>
        <i className="swatch is-offpeak" /> Off-peak {money(d.offpeak_cost)}
      </span>
      <span className="chart-tip-total">Total {money(d.total_cost)}</span>
    </div>
  );
}

function ForecastSection({ predicted, horizon, onHorizon }) {
  const p = predicted.data;
  const hourly = p?.breakdown_unit === "hour";
  const tick = (label) => (hourly ? hourLabel(Number(label.slice(11, 13))) : shortDay(label));

  return (
    <section className="panel panel-wide" aria-labelledby="forecast-title">
      <div className="panel-head">
        <div>
          <h2 id="forecast-title">Predicted cost</h2>
          <p className="panel-sub">Changing the period also updates the advice below.</p>
        </div>
        <HorizonSwitch value={horizon} onChange={onHorizon} />
      </div>

      {predicted.error && <Problem error={predicted.error} onRetry={predicted.reload} />}
      {!p && !predicted.error && <p className="panel-note">Pricing your forecast…</p>}
      {p && p.covered_hours === 0 && <p className="panel-note">There's no forecast for this period yet.</p>}

      {p && p.covered_hours > 0 && (
        <div className={predicted.loading ? "is-refreshing" : undefined}>
          <dl className="figures">
            <div>
              <dt>Total, {shortDay(p.period_start)} to {shortDay(p.period_end)}</dt>
              <dd>{money(p.predicted_total_cost, p.currency)}</dd>
            </div>
            <div>
              <dt>
                <i className="swatch is-peak" /> Peak
              </dt>
              <dd>{money(p.predicted_peak_cost, p.currency)}</dd>
            </div>
            <div>
              <dt>
                <i className="swatch is-offpeak" /> Off-peak
              </dt>
              <dd>{money(p.predicted_offpeak_cost, p.currency)}</dd>
            </div>
            <div>
              <dt>Energy</dt>
              <dd>{kwh(p.predicted_total_kwh)}</dd>
            </div>
          </dl>

          <div className="chart">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={p.breakdown} margin={{ top: 8, right: 4, left: -8, bottom: 0 }} barCategoryGap={hourly ? "12%" : "24%"}>
                <CartesianGrid stroke="#24344f" vertical={false} />
                <XAxis dataKey="label" tickFormatter={tick} stroke="#8a9bb5" fontSize={12} tickLine={false} axisLine={false} minTickGap={14} />
                <YAxis stroke="#8a9bb5" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(v) => `$${v}`} width={48} />
                <Tooltip content={<CostTooltip hourly={hourly} />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
                <Bar dataKey="offpeak_cost" stackId="c" fill={OFFPEAK} isAnimationActive={false} />
                <Bar dataKey="peak_cost" stackId="c" fill={PEAK} radius={[4, 4, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {p.is_partial_forecast && (
            <p className="panel-note">
              The forecast covers {Math.round(p.covered_hours)} of the {Math.round(p.expected_hours)} hours in this period,
              so the real cost will be higher.
            </p>
          )}
          <p className="source">
            {p.forecast.source === "cnn_lstm"
              ? "Usage forecast by the CNN-LSTM model trained on your uploaded data."
              : "No model forecast for this account yet, so this uses your average usage for each hour of the day over the last 8 weeks."}{" "}
            The period starts where your readings end.
          </p>
        </div>
      )}
    </section>
  );
}

function AdviceSection({ advice }) {
  const a = advice.data;
  return (
    <section className="panel panel-wide" aria-labelledby="advice-title">
      <div className="panel-head">
        <h2 id="advice-title">How to spend less</h2>
      </div>
      {advice.error && <Problem error={advice.error} onRetry={advice.reload} />}
      {!a && !advice.error && <p className="panel-note">Working out your recommendations…</p>}

      {a && (
        <div className={advice.loading ? "is-refreshing" : undefined}>
          {a.estimated_total_saving > 0 && (
            <div className="compare" role="img" aria-label={`From ${money(a.predicted_cost.predicted_total_cost)} to ${money(a.recommended_cost)}`}>
              <div className="compare-row">
                <span>As things are</span>
                <div className="compare-track">
                  <span className="compare-fill" style={{ width: "100%" }} />
                </div>
                <strong>{money(a.predicted_cost.predicted_total_cost)}</strong>
              </div>
              <div className="compare-row">
                <span>With these changes</span>
                <div className="compare-track">
                  <span
                    className="compare-fill is-better"
                    style={{ width: `${(a.recommended_cost / a.predicted_cost.predicted_total_cost) * 100}%` }}
                  />
                </div>
                <strong className="is-offpeak">{money(a.recommended_cost)}</strong>
              </div>
            </div>
          )}

          {a.recommendations.length === 0 ? (
            <p className="panel-note">Nothing to change right now: your predicted usage is already well timed.</p>
          ) : (
            <ol className="advice">
              {a.recommendations.map((r) => (
                <li key={r.rule} className={`advice-item is-${r.priority}`}>
                  <div className="advice-head">
                    <h3>{r.title}</h3>
                    <span className={`priority is-${r.priority}`}>{PRIORITY[r.priority] || r.priority}</span>
                  </div>
                  <p className="advice-do">{r.message}</p>
                  <dl className="advice-facts">
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
            </ol>
          )}
        </div>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------
export default function EnergyPage() {
  const [horizon, setHorizon] = useState("7day");
  const [version, setVersion] = useState(0); // bumped after saving rates so every figure recalculates
  const tariffQuery = useApi((token) => billingApi.getTariff(token), [version]);
  const tariff = tariffQuery.data?.found ? tariffQuery.data.tariff : null;
  const ready = Boolean(tariff);

  const cost = useApi((token) => billingApi.getCurrentCost(token), [version], ready);
  const predicted = useApi((token) => billingApi.getPredictedCost(token, horizon), [horizon, version], ready);
  const advice = useApi((token) => billingApi.getAdvice(token, horizon), [horizon, version], ready);

  const refresh = () => setVersion((v) => v + 1);

  let body;
  if (tariffQuery.loading && !tariffQuery.data) body = <p className="page-status">Loading your tariff…</p>;
  else if (tariffQuery.error) body = <Problem error={tariffQuery.error} onRetry={tariffQuery.reload} />;
  else if (!ready) body = <SetupHero onSaved={refresh} />;
  else
    body = (
      <>
        <Hero tariff={tariff} cost={cost} predicted={predicted} advice={advice} horizon={horizon} />
        <div className="panels">
          <TariffSection tariff={tariff} onSaved={refresh} />
          <CycleSection cost={cost} />
          <ForecastSection predicted={predicted} horizon={horizon} onHorizon={setHorizon} />
          <AdviceSection advice={advice} />
        </div>
      </>
    );

  return (
    <div className="energy">
      <header className="page-head">
        <h1>Energy</h1>
        <p>What your electricity costs, what it's likely to cost next, and how to bring it down.</p>
      </header>
      {isNoTariff(cost.error) ? <SetupHero onSaved={refresh} /> : body}
    </div>
  );
}