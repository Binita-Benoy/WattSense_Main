import { useCallback, useEffect, useMemo, useState } from "react";
import { useAuth } from "../context/AuthContext";
import { dashboardApi } from "../api/dashboardApi";
import AppHeader from "./AppHeader";
import StatCard from "./StatCard";
import EnergyChart from "./EnergyChart";
import HouseholdInfo from "./HouseholdInfo";
import DateRangeSelector, { shiftDate } from "./DateRangeSelector";
import AnomalyCheck from "./AnomalyCheck";
import { useLiveMeter, LiveControls, recalcHourly, liveHorizons } from "./LiveDemo";
import ForecastProgress from "./ForecastProgress";
import NewUserForecastChart from "./NewUserForecastChart";
import ForecastChart from "./ForecastChart";
import ForecastInfo from "./ForecastInfo";
import ReportButtons from "./ReportButtons";

const READINGS_LIMIT = 20000; // the most the API returns in one request


const USE_MOCK_FORECAST = false;
const kwhText = (value, digits) => (value == null ? "n/a" : `${Number(value).toFixed(digits)} kWh`);
// <<< END ADDED


export default function ExistingDashboard() {
  const { token } = useAuth();

  const [summary, setSummary] = useState(null); // household id + first/last reading dates
  const [dateRange, setDateRange] = useState(null);
  const [readings, setReadings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [reloadKey, setReloadKey] = useState(0);

  
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

 
  const stats = useMemo(() => {
    if (liveReadings.length === 0) return null;
    const values = liveReadings.map((r) => r.consumption_kwh ?? 0);
    const total = values.reduce((a, b) => a + b, 0);
    return { count: values.length, avg: total / values.length, peak: Math.max(...values), total };
  }, [liveReadings]);
 

  
  const [horizon, setHorizon] = useState("24h");
  const [forecast, setForecast] = useState(null);
  const [forecastError, setForecastError] = useState(null);

  useEffect(() => {
   
    if (!USE_MOCK_FORECAST || !dateRange || readings.length === 0) return undefined;
   
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

 
  const shownForecast = useMemo(() => {
    if (!forecast) return null;
    if (!meter.live.length) return forecast.predictions;
    return recalcHourly(meter.series, forecast.predictions.length).map((p) => ({
      timestamp: p.ts,
      predicted_kwh: p.kwh,
    }));
  }, [forecast, meter.live.length, meter.series]);
  
  const [job, setJob] = useState(null);
  const [jobError, setJobError] = useState(null);

  const refreshJob = useCallback(async () => {
    try {
      setJob(await dashboardApi.getNewUserForecast(token));
    } catch (err) {
      setJobError(err.message);
    }
  }, [token]);

  useEffect(() => {
    if (USE_MOCK_FORECAST) return undefined;
    let cancelled = false;
    (async () => {
      try {
        let j = await dashboardApi.getNewUserForecast(token);
        if (cancelled) return;
        if (j.status === "none") {
          await dashboardApi.startMyHouseholdForecast(token); // first visit: build it
          j = await dashboardApi.getNewUserForecast(token);
        }
        if (!cancelled) setJob(j);
      } catch (err) {
        if (!cancelled) setJobError(err.message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [token, reloadKey]);

  useEffect(() => {
    if (!job || (job.status !== "queued" && job.status !== "running")) return undefined;
    const id = setInterval(refreshJob, 3000);
    return () => clearInterval(id);
  }, [job?.status, refreshJob]);

  const retryForecast = async () => {
    setJobError(null);
    try {
      await dashboardApi.startMyHouseholdForecast(token, true);
      await refreshJob();
    } catch (err) {
      setJobError(err.message);
    }
  };

  const jobDone = job && job.status === "done" && job.result;
  const savedHorizons = useMemo(() => (jobDone ? job.result.horizons || {} : {}), [jobDone, job]);
  
  const atLatest = !!(summary && dateRange && dateRange.endDate === summary.last_reading.slice(0, 10));
  const demoOn = atLatest && meter.live.length > 0;
  const horizons = useMemo(
    () => liveHorizons(savedHorizons, meter.series, demoOn),
    [savedHorizons, meter.series, demoOn]
  );
  const forecastHistory = useMemo(() => {
    if (!jobDone) return [];
    return demoOn
      ? [...job.history, ...meter.live.map((p) => ({ timestamp: p.ts, kwh: p.kwh }))]
      : job.history;
  }, [jobDone, job, demoOn, meter.live]);
  const current = horizons[horizon];
  const modelText = current && !current.error ? `${current.method} (fine-tuned on this household's data)` : null;


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

          
          <EnergyChart readings={liveReadings} />
         
          {USE_MOCK_FORECAST ? (
            <>
              <ForecastInfo horizon={horizon} onHorizonChange={setHorizon} model={forecast?.model} />
              {forecastError && <div className="status error">{forecastError}</div>}
              {forecast && <ForecastChart readings={liveReadings} forecast={shownForecast} />}
            </>
          ) : (
            <>
              {jobError && (
                <div className="status error">
                  <p>{jobError}</p>
                  <button onClick={retryForecast}>Try again</button>
                </div>
              )}
              {!jobError && !job && <div className="status">Preparing your forecast…</div>}
              {job && (job.status === "queued" || job.status === "running") && <ForecastProgress job={job} />}
              {job && job.status === "failed" && (
                <div className="status error">
                  <p>{job.error || "The forecast failed."}</p>
                  <button onClick={retryForecast}>Try again</button>
                </div>
              )}
              {jobDone && (
                <>
                
                  <div className="household-info">
                    Forecast for <strong>{(job.result.location && job.result.location.resolved_name) || job.location}</strong>
                  </div>
                 
                  <div className="stat-grid">
                    <StatCard label="Next 24 hours" value={kwhText(horizons["24h"]?.total_kwh, 1)} />
                    <StatCard label="Next 7 days" value={kwhText(horizons["7day"]?.total_kwh, 1)} />
                    <StatCard label="Next 30 days" value={kwhText(horizons["1month"]?.total_kwh, 1)} />
                  </div>
                 
                  <ForecastInfo horizon={horizon} onHorizonChange={setHorizon} model={null} />
                 
                  {atLatest ? (
                    <LiveControls meter={meter} />
                  ) : (
                    <div className="household-info">
                      Live meter demo: set the end date to the latest date ({summary.last_reading.slice(0, 10)}) to use it.
                    </div>
                  )}
                  
                  <NewUserForecastChart history={forecastHistory} horizon={horizon} horizonData={current} />
                 
                </>
              )}
            </>
          )}
          

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