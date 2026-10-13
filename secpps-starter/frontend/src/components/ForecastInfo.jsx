const HORIZON_OPTIONS = [
  { value: "24h", label: "Next 24 hours" },
  { value: "7day", label: "Next 7 days" },
  { value: "1month", label: "Next month" },
];

export default function ForecastInfo({ horizon, onHorizonChange, model }) {
  return (
    <div className="forecast-info">
      <div className="forecast-info-row">
        <label htmlFor="horizon-select">Forecast horizon</label>
        <select
          id="horizon-select"
          value={horizon}
          onChange={(e) => onHorizonChange(e.target.value)}
        >
          {HORIZON_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>
      {model && (
        <div className="forecast-info-row">
          <span className="forecast-model-badge">Model: {model}</span>
        </div>
      )}
    </div>
  );
}