import { useEffect, useState } from "react";
import { api } from "./api";
import HouseholdSelector from "./components/HouseholdSelector";
import StatCard from "./components/StatCard";
import EnergyChart from "./components/EnergyChart";

export default function App() {
  const [households, setHouseholds] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [summary, setSummary] = useState(null);
  const [readings, setReadings] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  // Load the household list once on mount
  useEffect(() => {
    api
      .getHouseholds()
      .then((data) => {
        setHouseholds(data);
        if (data.length > 0) setSelectedId(data[0].household_id);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  // Load summary + readings whenever the selected household changes
  useEffect(() => {
    if (selectedId == null) return;
    setError(null);
    Promise.all([api.getSummary(selectedId), api.getReadings(selectedId)])
      .then(([summaryData, readingsData]) => {
        setSummary(summaryData);
        setReadings(readingsData);
      })
      .catch((err) => setError(err.message));
  }, [selectedId]);

  return (
    <div className="app">
      <div className="header">
        <h1>SECPPS Energy Dashboard</h1>
        <p>Smart energy consumption prediction &amp; optimisation — live view</p>
      </div>

      {loading && <div className="status">Loading households…</div>}
      {error && <div className="status error">{error}</div>}

      {!loading && !error && households.length > 0 && (
        <>
          <HouseholdSelector
            households={households}
            selectedId={selectedId}
            onChange={setSelectedId}
          />

          <div className="stat-grid">
            <StatCard label="Readings" value={summary?.reading_count} />
            <StatCard
              label="Avg kWh"
              value={summary?.avg_kwh !== undefined ? `${summary.avg_kwh}` : undefined}
            />
            <StatCard
              label="Peak kWh"
              value={summary?.max_kwh !== undefined ? `${summary.max_kwh}` : undefined}
            />
            <StatCard
              label="Total kWh"
              value={summary?.total_kwh !== undefined ? `${summary.total_kwh}` : undefined}
            />
          </div>

          {readings.length > 0 && <EnergyChart readings={readings} />}
        </>
      )}

      {!loading && !error && households.length === 0 && (
        <div className="status">No households found in the database yet.</div>
      )}
    </div>
  );
}