import { useState } from "react";
import { useAuth } from "../context/AuthContext";
import { dashboardApi } from "../api/dashboardApi";

const MAX_BYTES = 15 * 1024 * 1024;

/** New users upload a CSV of their past hourly usage plus their location. */
export default function UploadForm({ onStarted }) {
  const { token } = useAuth();
  const [file, setFile] = useState(null);
  const [location, setLocation] = useState("");
  const [countryCode, setCountryCode] = useState("AU");
  const [state, setState] = useState("NSW");
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const handleFile = (e) => {
    const picked = e.target.files && e.target.files[0] ? e.target.files[0] : null;
    setError(null);
    if (picked && picked.size > MAX_BYTES) {
      setFile(null);
      setError("That file is larger than 15 MB.");
      return;
    }
    setFile(picked);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please choose your CSV file first.");
      return;
    }
    if (location.trim().length < 2) {
      setError("Please enter your location, e.g. Sydney, Australia.");
      return;
    }
    setError(null);
    setSubmitting(true);
    try {
      const response = await dashboardApi.startNewUserForecast(token, {
        file,
        location: location.trim(),
        countryCode: countryCode.trim().toUpperCase() || "AU",
        state: state.trim().toUpperCase(),
      });
      onStarted(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="chart-panel">
      <h2>Upload your electricity usage</h2>
      <p className="hint">
        Upload your past usage (ideally a full year, ending recently) and your location. We train
        the forecasting model on your data and show your forecast here.
      </p>

      <form className="upload-form" onSubmit={handleSubmit}>
        <div className="field">
          <label htmlFor="csv-file">Usage file (CSV)</label>
          <input id="csv-file" type="file" accept=".csv,text/csv" onChange={handleFile} />
          {file && <span className="hint">{file.name} ({(file.size / 1024).toFixed(0)} KB)</span>}
        </div>

        <div className="field">
          <label htmlFor="location">Location</label>
          <input
            id="location"
            type="text"
            placeholder="e.g. Sydney, Australia"
            value={location}
            onChange={(e) => setLocation(e.target.value)}
          />
        </div>

        <div className="field-row">
          <div className="field">
            <label htmlFor="country">Country code</label>
            <input
              id="country"
              type="text"
              maxLength={2}
              value={countryCode}
              onChange={(e) => setCountryCode(e.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor="state">State (for public holidays)</label>
            <input
              id="state"
              type="text"
              maxLength={8}
              value={state}
              onChange={(e) => setState(e.target.value)}
            />
          </div>
        </div>

        {error && <div className="status error" role="alert">{error}</div>}

        <button type="submit" className="primary-button" disabled={submitting}>
          {submitting ? "Uploading..." : "Upload and build my forecast"}
        </button>
      </form>

      <div className="format-help">
        <strong>File format</strong>
        <ul>
          <li>Two columns named <code>timestamp</code> and <code>kwh</code></li>
          <li>One reading per hour. 30-minute readings are added up into hours for you</li>
          <li>At least 14 days; about 38+ days for the 1-month forecast; a full year works best</li>
        </ul>
        <pre className="code-example">{`timestamp,kwh
2025-10-05 00:00,0.32
2025-10-05 01:00,0.16`}</pre>
      </div>
    </div>
  );
}