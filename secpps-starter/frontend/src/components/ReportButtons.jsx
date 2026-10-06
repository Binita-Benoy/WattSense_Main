import { useState } from "react";
import { dashboardApi } from "../api/dashboardApi";

/**
 * CSV / PDF report. `source`: "household" (dataset household, uses `range`) or "upload" (saved upload).
 * Cost is only included if the user types their own rate: we don't have their tariff.
 */
export default function ReportButtons({ token, source = "household", range }) {
  const [rate, setRate] = useState("");
  const [downloading, setDownloading] = useState(null);
  const [error, setError] = useState(null);

  const handleDownload = async (format) => {
    setError(null);
    setDownloading(format);
    try {
      const parsed = parseFloat(rate);
      await dashboardApi.downloadReport(token, format, {
        source,
        ...range,
        ratePerKwh: parsed > 0 ? parsed : undefined,
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setDownloading(null);
    }
  };

  return (
    <div className="chart-panel">
      <h2>Reports</h2>
      <div className="report-rate">
        <label htmlFor="rate">Your electricity rate ($ per kWh), optional</label>
        <input
          id="rate"
          type="number"
          step="0.01"
          min="0"
          placeholder="e.g. 0.30"
          value={rate}
          onChange={(e) => setRate(e.target.value)}
        />
        <span className="hint">Enter it to include cost figures in the report.</span>
      </div>
      <div className="report-buttons">
        <button onClick={() => handleDownload("csv")} disabled={downloading === "csv"}>
          {downloading === "csv" ? "Generating..." : "Download CSV"}
        </button>
        <button onClick={() => handleDownload("pdf")} disabled={downloading === "pdf"}>
          {downloading === "pdf" ? "Generating..." : "Download PDF"}
        </button>
      </div>
      {error && <div className="status error">{error}</div>}
    </div>
  );
}