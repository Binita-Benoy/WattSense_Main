# data_pipeline — owner: Alka

Responsible for: ingestion (MQTT/REST/CSV), cleaning, resampling, storage into
`energy_readings` / `weather_readings` (see `infra/init-db/01-init.sql`).

## Suggested first files
- `ingest_csv.py` — load a SGSC CSV slice into Postgres via `households` +
  `energy_readings`
- `clean.py` — interpolation, 3-sigma outlier removal, hourly resampling
- `weather.py` — Open-Meteo API pull, writes to `weather_readings`

## Output contract (must not change without telling Mishael + Binita)
A cleaned, hourly-resampled row looks like:

```json
{
  "household_id": "H0001",
  "ts": "2026-08-19T14:00:00+10:00",
  "kwh": 1.243
}
```
