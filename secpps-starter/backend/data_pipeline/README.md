# Data Engineering Pipeline

**Owner:** Alka Shilpakar  
**Project:** WattSense / SECPPS

This directory contains the data engineering pipeline for the WattSense smart energy platform.

The pipeline currently processes Australian Smart Grid Smart City (SGSC) household energy data, stores the cleaned hourly readings in PostgreSQL, retrieves historical weather data from Open-Meteo, and integrates the energy and weather datasets for downstream machine learning and forecasting.

---

## 1. Current Pipeline Status

The following components have been implemented:

- SGSC smart-meter dataset profiling
- Selection of 5 diverse household profiles
- SGSC data extraction from the full interval-reading dataset
- Missing timestamp reconstruction
- Time-aware interpolation
- Rolling 3-sigma outlier treatment
- 30-minute to hourly energy resampling
- Data validation
- PostgreSQL ingestion
- Open-Meteo historical weather retrieval
- Weather-to-household integration
- Solar irradiance (`shortwave_radiation`) integration
- Energy/weather timestamp validation
- Docker integration
- Reproducible PostgreSQL database dump

Current database contents:

| Dataset | Records |
|---|---:|
| SGSC households | 5 |
| SGSC hourly energy readings | 104,593 |
| Open-Meteo weather readings | 105,005 |

Energy-to-weather matching has been validated with **0 missing weather matches across all 104,593 SGSC energy readings**.

HomeC is **not used in the current pipeline or database**.

---

## 2. Directory Structure

```text
backend/data_pipeline/
├── README.md
├── clean.py
├── ingest_csv.py
└── weather.py
```

### `clean.py`

Cleans and transforms the extracted SGSC interval-reading dataset.

Main operations:

1. Parse SGSC timestamps.
2. Convert energy columns to numeric values.
3. Remove duplicate household/timestamp readings.
4. Reconstruct a continuous 30-minute time index for each household.
5. Apply time-aware interpolation to missing energy readings.
6. Detect values outside a rolling 3-sigma range using a 48-reading (24-hour) window.
7. Replace detected outliers with missing values and interpolate them.
8. Resample 30-minute kWh readings to hourly kWh using `SUM`.
9. Validate the resulting dataset.
10. Save the cleaned hourly dataset.

Input:

```text
data/interim/sgsc_selected_households.csv
```

Output:

```text
data/processed/sgsc_hourly_clean.csv
```

The processed dataset contains:

```text
104,593 hourly readings
5 households
0 invalid timestamps
0 duplicate household/timestamp combinations
0 missing GENERAL_SUPPLY_KWH values
0 negative GENERAL_SUPPLY_KWH values
0 non-hour-aligned timestamps
```

---

## 3. Selected SGSC Households

Five households were selected to provide variation in electricity-use level, dwelling type, climate zone, and air-conditioning characteristics.

| Household ID | Electricity Use | Dwelling Type | Climate Zone | Air Conditioning |
|---|---|---|---|---|
| 10042974 | MED | Unit | Mild temperate | Yes |
| 10346126 | MED | Separate house | Mild temperate | Yes |
| 10733884 | LOW | Separate house | Mild temperate | Yes |
| 10859314 | MED | Semi-detached | Warm temperate | No |
| 11228008 | HIGH | Separate house | Warm temperate | No |

The selected households currently do not provide solar-generation profiles.

---

## 4. SGSC Energy Preprocessing

The original SGSC interval dataset contains approximately:

```text
135,338,336 readings
5,332 customers
```

Because the raw CSV is several GB in size, it was processed using chunked reading rather than loading the entire dataset into memory.

The five selected households produced:

```text
208,545 selected raw interval readings
```

After preprocessing and hourly resampling:

```text
104,593 clean hourly energy readings
```

Hourly row counts:

| Household ID | Hourly Rows |
|---|---:|
| 10042974 | 20,999 |
| 10346126 | 20,950 |
| 10733884 | 20,978 |
| 10859314 | 20,879 |
| 11228008 | 20,787 |
| **Total** | **104,593** |

---

## 5. Loading SGSC Data into PostgreSQL

`ingest_csv.py` loads the cleaned hourly SGSC dataset into PostgreSQL.

It:

- validates required columns;
- checks timestamps;
- checks duplicate household/timestamp records;
- checks for negative consumption;
- inserts SGSC households;
- loads energy readings in batches;
- uses conflict handling to avoid duplicate energy readings;
- verifies database row counts after ingestion.

Run inside Docker:

```bash
docker compose exec backend python data_pipeline/ingest_csv.py
```

Expected final verification:

```text
SGSC households:      5
SGSC energy readings: 104,593

SGSC ingestion completed successfully.
```

---

## 6. Open-Meteo Weather Integration

`weather.py` retrieves historical weather data corresponding to the SGSC energy period.

The current SGSC energy range is:

```text
2011-10-10 10:00:00
to
2014-03-03 10:00:00
```

The weather pipeline uses the SGSC dataset-level geographic reference:

```text
Latitude:  -32.92921
Longitude: 151.77002
Timezone:  Australia/Sydney
```

This is a common reference location for the prototype. Individual household coordinates are not available in the current dataset, so household-specific locations are not fabricated.

The following hourly weather variables are retrieved:

| Open-Meteo Variable | Database Column |
|---|---|
| `temperature_2m` | `temperature` |
| `relative_humidity_2m` | `humidity` |
| `dew_point_2m` | `dew_point` |
| `apparent_temperature` | `apparent_temperature` |
| `precipitation` | `precip_intensity` |
| `pressure_msl` | `pressure` |
| `cloud_cover` | `cloud_cover` |
| `wind_speed_10m` | `wind_speed` |
| `wind_direction_10m` | `wind_bearing` |
| `shortwave_radiation` | `shortwave_radiation` |

`shortwave_radiation` provides the solar-radiation/weather feature required by the project.

The common weather series is associated with each of the five selected SGSC households to simplify downstream household/timestamp joins.

Run:

```bash
docker compose exec backend python data_pipeline/weather.py
```

Current result:

```text
Weather rows per household: 21,001
Households:                  5
Total weather rows:          105,005
```

Validation results:

```text
Duplicate weather timestamps:       0
Missing temperature values:         0
Missing humidity values:            0
Missing shortwave radiation values: 0
```

---

## 7. Energy + Weather Integration

Energy and weather data are joined using:

```text
household_id
+
reading_timestamp
```

Validation was performed using a left join between `energy_readings` and `weather_readings`.

Results:

| Household | Energy Rows | Weather Matches | Missing Weather |
|---|---:|---:|---:|
| 10042974 | 20,999 | 20,999 | 0 |
| 10346126 | 20,950 | 20,950 | 0 |
| 10733884 | 20,978 | 20,978 | 0 |
| 10859314 | 20,879 | 20,879 | 0 |
| 11228008 | 20,787 | 20,787 | 0 |
| **Total** | **104,593** | **104,593** | **0** |

Therefore, all current SGSC energy readings have corresponding weather information.

---

# Database Setup

## 8. PostgreSQL with Docker

The development database runs through Docker Compose.

The current database service uses:

```text
PostgreSQL 16
Database: smart_energy
User: secpps
Container: secpps_db
```

Start the project from the repository root:

```bash
docker compose up --build -d
```

Check the containers:

```bash
docker compose ps
```

Expected services include:

```text
secpps_db
secpps_backend
secpps_frontend
```

The database should report a healthy status.

---

## 9. Backend Database Connection

Inside Docker, the backend connects to PostgreSQL using:

```text
postgresql://secpps:secpps_dev_password@db:5432/smart_energy
```

This is configured through the `DATABASE_URL` environment variable in `docker-compose.yml`.

Important:

```text
db:5432
```

is the Docker-internal database address.

Do not replace `db` with `localhost` when code is running inside the backend container.

From the host machine, PostgreSQL is exposed on:

```text
localhost:5432
```

provided another local PostgreSQL server is not already using that port.

---

## 10. Connecting to PostgreSQL

To open PostgreSQL directly inside the database container:

```bash
docker compose exec db psql -U secpps -d smart_energy
```

Useful PostgreSQL commands:

```sql
\dt
```

Lists tables.

```sql
\d energy_readings
```

Shows the `energy_readings` schema.

```sql
\d weather_readings
```

Shows the weather schema.

Exit with:

```sql
\q
```

---

## 11. Checking Database Contents

SGSC households:

```bash
docker compose exec db psql -U secpps -d smart_energy -c "
SELECT source, COUNT(*)
FROM households
GROUP BY source;
"
```

Energy readings:

```bash
docker compose exec db psql -U secpps -d smart_energy -c "
SELECT source, COUNT(*)
FROM energy_readings
GROUP BY source;
"
```

Weather readings:

```bash
docker compose exec db psql -U secpps -d smart_energy -c "
SELECT source, COUNT(*)
FROM weather_readings
GROUP BY source;
"
```

Current expected values:

```text
households:
SGSC          5

energy_readings:
SGSC          104593

weather_readings:
Open-Meteo    105005
```

---

# Database Dump

## 12. What is `smart_energy.dump`?

A PostgreSQL database dump is included at:

```text
infra/init-db/smart_energy.dump
```

The dump provides a reproducible snapshot of the populated project database.

The current dump contains:

```text
5 SGSC households
104,593 SGSC hourly energy readings
105,005 Open-Meteo weather readings
```

It also contains the PostgreSQL table structures, constraints, indexes, sequences, and the `shortwave_radiation` weather field.

HomeC data has been removed and is not part of the current database snapshot.

---

## 13. Automatic Dump Restore

The repository contains:

```text
infra/init-db/02-restore.sh
```

The PostgreSQL Docker container executes initialization scripts from:

```text
/docker-entrypoint-initdb.d/
```

The restore script automatically restores:

```text
smart_energy.dump
```

when PostgreSQL initializes a **new empty Docker volume**.

Important: PostgreSQL initialization scripts run only when the database volume is first created. They do not run every time the container restarts.

Therefore:

```bash
docker compose stop
docker compose start
```

does not restore the database again.

---

## 14. Manually Restoring the Dump

If a manual restore is required, create an empty database and restore the dump using `pg_restore`.

Example:

```bash
docker compose exec db createdb -U secpps smart_energy_restore
```

Then:

```bash
docker compose exec -T db pg_restore \
  -U secpps \
  -d smart_energy_restore \
  --no-owner \
  < infra/init-db/smart_energy.dump
```

Check it:

```bash
docker compose exec db psql \
  -U secpps \
  -d smart_energy_restore \
  -c "\dt"
```

---

## 15. Creating an Updated Dump

After intentionally changing the database and verifying the changes, regenerate the dump with:

```bash
docker compose exec -T db pg_dump \
  -U secpps \
  -d smart_energy \
  -Fc \
  --no-owner \
  > infra/init-db/smart_energy.dump
```

This overwrites the existing dump with the current state of `smart_energy`.

Do not regenerate the shared dump from an unverified development database.

---

## 16. Verifying the Dump

List its contents:

```bash
pg_restore -l infra/init-db/smart_energy.dump
```

Check specifically for weather data:

```bash
pg_restore -l infra/init-db/smart_energy.dump | grep "weather_readings"
```

A populated dump should include:

```text
TABLE DATA public weather_readings
```

For stronger verification, restore the dump into a temporary database:

```bash
docker compose exec db createdb -U secpps smart_energy_verify
```

```bash
docker compose exec -T db pg_restore \
  -U secpps \
  -d smart_energy_verify \
  --no-owner \
  < infra/init-db/smart_energy.dump
```

Then verify:

```bash
docker compose exec db psql -U secpps -d smart_energy_verify -c "
SELECT source, COUNT(*) FROM households GROUP BY source;
SELECT source, COUNT(*) FROM energy_readings GROUP BY source;
SELECT source, COUNT(*) FROM weather_readings GROUP BY source;
"
```

Expected:

```text
SGSC         5
SGSC         104593
Open-Meteo   105005
```

After verification:

```bash
docker compose exec db dropdb -U secpps smart_energy_verify
```

---

# Running the Data Pipeline

## 17. Pipeline Order

When rebuilding the data from source rather than using the database dump, run the pipeline in this order:

### Step 1 — Clean SGSC

```bash
docker compose exec backend python data_pipeline/clean.py
```

Produces:

```text
/data/processed/sgsc_hourly_clean.csv
```

### Step 2 — Load SGSC into PostgreSQL

```bash
docker compose exec backend python data_pipeline/ingest_csv.py
```

### Step 3 — Retrieve and Load Weather

```bash
docker compose exec backend python data_pipeline/weather.py
```

### Step 4 — Validate Database

Check energy:

```sql
SELECT source, COUNT(*)
FROM energy_readings
GROUP BY source;
```

Check weather:

```sql
SELECT source, COUNT(*)
FROM weather_readings
GROUP BY source;
```

---

# Downstream Usage

## 18. Data Available to ML and Forecasting

The data engineering pipeline provides clean hourly household energy data and aligned historical weather data.

A downstream query can join the two datasets with:

```sql
SELECT
    e.household_id,
    e.reading_timestamp,
    e.consumption_kwh,
    e.controlled_load_kwh,
    e.gross_generation_kwh,
    e.net_generation_kwh,
    w.temperature,
    w.humidity,
    w.apparent_temperature,
    w.pressure,
    w.wind_speed,
    w.cloud_cover,
    w.precip_intensity,
    w.dew_point,
    w.shortwave_radiation
FROM energy_readings e
JOIN weather_readings w
    ON e.household_id = w.household_id
   AND e.reading_timestamp = w.reading_timestamp
WHERE e.source = 'SGSC'
  AND w.source = 'Open-Meteo'
ORDER BY e.household_id, e.reading_timestamp;
```

This integrated hourly dataset can then be used by the ML/forecasting components for feature engineering, anomaly detection, energy forecasting, and related modelling.

Feature engineering and model implementation are handled by the corresponding ML/AI components rather than inside this data pipeline.

---

## 19. Important Notes

- The current Docker database is PostgreSQL 16.
- The current implementation does not use TimescaleDB-specific functionality.
- SGSC household IDs are stored as their original numeric IDs.
- The database currently uses `TIMESTAMP WITHOUT TIME ZONE`.
- Open-Meteo is requested using `Australia/Sydney`.
- The Open-Meteo location is a common SGSC dataset-level reference point, not an individual household location.
- HomeC is no longer part of the active data pipeline.
- Raw and intermediate datasets should not be committed to Git unless explicitly required.
- `smart_energy.dump` is the reproducible populated database snapshot for the team.