const PRESETS = [
  { label: "Last 7 days", days: 7 },
  { label: "Last 30 days", days: 30 },
  { label: "Last 90 days", days: 90 },
];

function isoDateDaysAgo(days) {
  const d = new Date();
  d.setDate(d.getDate() - days);
  return d.toISOString().split("T")[0];
}

// Shifts a YYYY-MM-DD date by N days (UTC maths, so no off-by-one from timezones)
function shiftDate(iso, days) {
  const d = new Date(`${iso}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().split("T")[0];
}

// minDate / maxDate = first / last day that has data. Presets count back from maxDate
// (not from today), because a household's data doesn't necessarily end today.
export default function DateRangeSelector({ startDate, endDate, onChange, minDate, maxDate }) {
  const latest = maxDate || isoDateDaysAgo(0);

  const applyPreset = (days) => {
    const start = shiftDate(latest, -days);
    onChange({
      startDate: minDate && start < minDate ? minDate : start,
      endDate: latest,
    });
  };

  return (
    <div className="controls">
      <div className="date-range-presets">
        {PRESETS.map((preset) => (
          <button key={preset.days} onClick={() => applyPreset(preset.days)}>
            {preset.label}
          </button>
        ))}
      </div>
      <div className="date-range-custom">
        <label>
          From:{" "}
          <input
            type="date"
            value={startDate}
            min={minDate}
            max={endDate}
            onChange={(e) => onChange({ startDate: e.target.value, endDate })}
          />
        </label>
        <label>
          To:{" "}
          <input
            type="date"
            value={endDate}
            min={startDate}
            max={latest}
            onChange={(e) => onChange({ startDate, endDate: e.target.value })}
          />
        </label>
      </div>
      {minDate && maxDate && (
        <span className="date-range-note">
          Data available: {minDate} to {maxDate}
        </span>
      )}
    </div>
  );
}

export { isoDateDaysAgo, shiftDate };