import { useEffect, useMemo, useState } from "react";
import { useAuth } from "../context/AuthContext";
import { dashboardApi } from "../api/dashboardApi";
import AppHeader from "./AppHeader";
import StatCard from "./StatCard";
import EnergyChart from "./EnergyChart";
import HouseholdInfo from "./HouseholdInfo";
import DateRangeSelector, { shiftDate } from "./DateRangeSelector";
import AnomalyCheck from "./AnomalyCheck";
// >>> ADDED: live meter demo
import { useLiveMeter, LiveControls, recalcHourly } from "./LiveDemo";
// <<< END ADDED
import ForecastChart from "./ForecastChart";
import ForecastInfo from "./ForecastInfo";
import ReportButtons from "./ReportButtons";

const READINGS_LIMIT = 20000; // the most the API returns in one request

/**
 * Dashboard for "existing users": any account that was not registered as a new user, shown its dataset household.
 * (This is the dashboard that used to live inside App.jsx.) Fixes compared with before:
 *   - the date range now actually filters the data
 *   - the default range is the last 30 days OF THE HOUSEHOLD'S DATA, not 30 days before today
 *   - the stat cards are worked out from the readings in the selected range
 *   - the forecast starts where the selected range ends, so there is no gap in the chart
 */
export default function ExistingDashboard() {
  const { token } = useAuth();

  const [summary, setSummary] = useState(null); // household id + first/last reading dates
  const [dateRange, setDateRange] = useState(null);
  const [readings, setReadings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [reloadKey, setReloadKey] = useState(0);

  // >>> ADDED: live meter demo. Extra readings (one per 30 s) are appended to `readings`, so the existing graphs,
  // stat cards and forecast all update. `readings` itself (the real API data) is left untouched.
  const baseSeries = useMemo(
    () => readings.map((r) => ({ ts: r.reading_timestamp, kwh: r.consumption_kwh ?? 0 })),
    [readings]
  );
  const meter = useLiveMeter(baseSeries);
  const liveReadings = useMemo(
    () =>
      meter.live.length
        ? [...readings, ...meter.live.map((p) => ({ reading_timestamp: p.ts, consumption_kwh: p.kwh }))]
        : readings,
    [readings, meter.live]
  );
  // <<< END ADDED

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    dashboardApi
      .getMySummary(token)
      .then((s) => {
        if (cancelled) return;
        setSummary(s);
        const first = s.first_reading.slice(0, 10);
        const last = s.last_reading.slice(0, 10);
        const start = shiftDate(last, -30);
        setDateRange({ startDate: start < first ? first : start, endDate: last });
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err.message);
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [token, reloadKey]);

  useEffect(() => {
    if (!dateRange) return undefined;
    let cancelled = false;
    setLoading(true);
    setError(null);
    dashboardApi
      .getMyReadings(token, {
        start: `${dateRange.startDate}T00:00:00`,
        end: `${dateRange.endDate}T23:59:59`,
        limit: READINGS_LIMIT,
      })
      .then((r) => {
        if (!cancelled) setReadings(r);
      })
      .catch((err) => {
        if (cancelled) return;
        setReadings([]);
        setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [token, dateRange?.startDate, dateRange?.endDate, reloadKey]);

  // >>> CHANGED: stats use liveReadings (real readings + demo readings). Before: readings
  const stats = useMemo(() => {
    if (liveReadings.length === 0) return null;
    const values = liveReadings.map((r) => r.consumption_kwh ?? 0);
    const total = values.reduce((a, b) => a + b, 0);
    return { count: values.length, avg: total / values.length, peak: Math.max(...values), total };
  }, [liveReadings]);
  // <<< END CHANGED

  // --- forecast (mock for now; see api/forecast.py) -------------------------
  const [horizon, setHorizon] = useState("24h");
  const [forecast, setForecast] = useState(null);
  const [forecastError, setForecastError] = useState(null);

  useEffect(() => {
    if (!dateRange || readings.length === 0) return undefined;
    let cancelled = false;
    setForecastError(null);
    dashboardApi
      .getMyForecast(token, horizon, `${dateRange.endDate}T23:59:59`)
      .then((f) => {
        if (!cancelled) setForecast(f);
      })
      .catch((err) => {
        if (!cancelled) setForecastError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [token, horizon, dateRange?.endDate, readings.length]);

  // >>> ADDED: once live readings exist, the forecast is recalculated from them (same length as the real forecast)
  const shownForecast = useMemo(() => {
    if (!forecast) return null;
    if (!meter.live.length) return forecast.predictions;
    return recalcHourly(meter.series, forecast.predictions.length).map((p) => ({
      timestamp: p.ts,
      predicted_kwh: p.kwh,
    }));
  }, [forecast, meter.live.length, meter.series]);
  // <<< END ADDED

  return (
    <div className="app">
      <AppHeader subtitle="Smart energy consumption prediction & optimisation — live view" />

      {summary && <HouseholdInfo householdId={summary.household_id} />}

      {dateRange && summary && (
        <DateRangeSelector
          startDate={dateRange.startDate}
          endDate={dateRange.endDate}
          minDate={summary.first_reading.slice(0, 10)}
          maxDate={summary.last_reading.slice(0, 10)}
          onChange={setDateRange}
        />
      )}

      {loading && <div className="status">Loading your consumption data…</div>}

      {error && !loading && (
        <div className="status error">
          <p>{error}</p>
          <button onClick={() => setReloadKey((k) => k + 1)}>Try again</button>
        </div>
      )}

      {!loading && !error && summary && stats && (
        <>
          <div className="stat-grid">
            <StatCard label="Readings in range" value={stats.count} />
            <StatCard label="Avg kWh" value={stats.avg.toFixed(3)} />
            <StatCard label="Peak kWh" value={stats.peak.toFixed(3)} />
            <StatCard label="Total kWh" value={stats.total.toFixed(2)} />
          </div>

          {readings.length >= READINGS_LIMIT && (
            <div className="info-note">
              Showing the first {READINGS_LIMIT.toLocaleString()} readings of this range. Narrow the
              date range to see the rest.
            </div>
          )}

          {/* >>> ADDED: live meter controls (sit right above the first graph) */}
          <LiveControls meter={meter} />
          {/* <<< END ADDED */}

          {/* >>> CHANGED: graph shows liveReadings. Before: readings={readings} */}
          <EnergyChart readings={liveReadings} />
          {/* <<< END CHANGED */}

          <ForecastInfo horizon={horizon} onHorizonChange={setHorizon} model={forecast?.model} />
          {forecastError && <div className="status error">{forecastError}</div>}
          {/* >>> CHANGED: uses liveReadings / shownForecast. Before: readings={readings} forecast={forecast.predictions} */}
          {forecast && <ForecastChart readings={liveReadings} forecast={shownForecast} />}
          {/* <<< END CHANGED */}

          <AnomalyCheck token={token} />

          <ReportButtons
            token={token}
            source="household"
            range={{
              start: `${dateRange.startDate}T00:00:00`,
              end: `${dateRange.endDate}T23:59:59`,
            }}
          />
        </>
      )}
    </div>
  );
}