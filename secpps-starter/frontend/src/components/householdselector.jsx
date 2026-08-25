export default function HouseholdSelector({ households, selectedId, onChange }) {
  return (
    <div className="controls">
      <label htmlFor="household-select">Household</label>
      <select
        id="household-select"
        value={selectedId ?? ""}
        onChange={(e) => onChange(Number(e.target.value))}
      >
        {households.map((h) => (
          <option key={h.household_id} value={h.household_id}>
            #{h.household_id} — {h.dwelling_type || "unknown dwelling"}
            {h.climate_zone ? ` · ${h.climate_zone}` : ""}
          </option>
        ))}
      </select>
    </div>
  );
}