# Data Engineering Pipeline

**Owner:** Alka Shilpakar  
**Project:** WattSense / SECPPS

This directory contains the data engineering pipeline for the WattSense smart energy platform.

The pipeline processes Australian Smart Grid Smart City (SGSC) household energy data, selects and extracts 30 household profiles, preprocesses the interval readings into continuous hourly energy data, stores the results in PostgreSQL, retrieves historical weather data from Open-Meteo, and integrates the energy and weather datasets for downstream machine learning and forecasting.

---

## 1. Current Pipeline Status

The following components have been implemented:

- SGSC smart-meter dataset profiling
- Quality-based selection of 30 household profiles
- Chunked SGSC extraction from the full interval-reading dataset
- Household metadata extraction and integration
- Duplicate reading removal
- Missing timestamp reconstruction
- Time-aware interpolation
- Rolling 3-sigma outlier treatment
- 30-minute to hourly energy resampling
- Data validation and continuity checking
- PostgreSQL ingestion
- Open-Meteo historical weather retrieval
- Weather-to-household integration
- Solar irradiance (`shortwave_radiation`) integration
- Energy/weather timestamp validation
- Docker integration
- Reproducible PostgreSQL database dump
- Database dump restore verification

Current database contents:

| Dataset | Records |
|---|---:|
| SGSC households | 30 |
| SGSC hourly energy readings | 625,424 |
| Open-Meteo weather readings | 630,120 |

Energy-to-weather matching has been validated with:

```text
Total SGSC energy readings: 625,424
Matched weather readings:   625,424
Missing weather readings:   0
Weather coverage:            100.00%
```

HomeC is **not used in the current pipeline or database**.

---

## 2. Directory Structure

```text
backend/data_pipeline/
├── README.md
├── extract_sgsc.py
├── clean.py
├── ingest_csv.py
└── weather.py
```

### `extract_sgsc.py`

Selects the 30 SGSC households and extracts their interval readings from the full SGSC interval dataset.

The full interval-reading CSV is several GB in size and is therefore processed in chunks rather than loaded completely into memory.

Outputs:

```text
data/interim/sgsc_selected_30_households.csv
data/interim/sgsc_selected_households.csv
```

### `clean.py`

Cleans and transforms the extracted SGSC interval-reading dataset.

Main operations:

1. Normalize SGSC CSV column names.
2. Parse SGSC timestamps.
3. Convert energy columns to numeric values.
4. Remove duplicate household/timestamp readings.
5. Reconstruct a continuous 30-minute time index for each household.
6. Apply time-aware interpolation to missing energy readings.
7. Detect values outside a rolling 3-sigma range using a 48-reading (24-hour) window.
8. Replace detected outliers with missing values and interpolate them.
9. Resample 30-minute kWh readings to hourly kWh using `SUM`.
10. Validate the resulting hourly dataset.
11. Save the cleaned dataset.

Input:

```text
data/interim/sgsc_selected_households.csv
```

Output:

```text
data/processed/sgsc_hourly_clean.csv
```

### `ingest_csv.py`

Loads the cleaned hourly energy data and selected household metadata into PostgreSQL.

### `weather.py`

Retrieves historical Open-Meteo weather data corresponding to the SGSC energy period and stores it in PostgreSQL.

---

## 3. SGSC Household Selection

The pipeline uses **30 SGSC households**.

The household-selection process was designed to provide a larger and more diverse dataset while maintaining data-quality requirements.

The original five prototype households were retained:

```text
10042974
10346126
10733884
10859314
11228008
```

Additional households were selected using SGSC customer summaries and household metadata.

Selection considered:

- available reading count;
- missing energy data;
- electricity-use group;
- dwelling type;
- climate zone;
- air-conditioning status;
- generation status; and
- metadata completeness.

A minimum reading-count threshold of:

```text
40,000 interval readings
```

was used for the final selection.

Households with missing energy values were excluded from the eligible selection pool.

The final electricity-use distribution is:

| Electricity Use Group | Households |
|---|---:|
| LOW | 10 |
| MED | 12 |
| HI | 8 |
| **Total** | **30** |

Climate-zone distribution:

| Climate Zone | Households |
|---|---:|
| Mild temperate | 19 |
| Warm temperate | 11 |
| **Total** | **30** |

Dwelling-type distribution:

| Dwelling Type | Households |
|---|---:|
| Separate house | 12 |
| Unit | 6 |
| Semi-detached | 2 |
| Not specified | 10 |
| **Total** | **30** |

Air-conditioning distribution:

| Air Conditioning | Households |
|---|---:|
| Yes | 22 |
| No | 8 |
| **Total** | **30** |

The final selected households do not contain generation profiles.

Generation households were considered during dataset analysis; however, none of the available generation households satisfied the final minimum threshold of 40,000 interval readings together with the required energy-data quality criteria.

The quality threshold was therefore retained rather than weakening the selection criteria solely to include generation profiles.

The 30 selected household IDs are:

```text
8162493
8163939
10042604
10042756
10042934
10042974
10174306
10257896
10305996
10341486
10346126
10346582
10390745
10413359
10508173
10508205
10534327
10534355
10673547
10673789
10733766
10733884
10760980
10850536
10859314
10883534
10885220
11006864
11228008
11446117
```

---

## 4. SGSC Extraction

The full SGSC interval dataset contains approximately:

```text
135,338,336 readings
5,332 customers
```

Because the source CSV is several GB in size, `extract_sgsc.py` processes it using chunked CSV reading.

Only records belonging to the selected household IDs are retained.

Final extraction validation:

```text
Selected households:     30
Extracted interval rows:  1,247,602
Duplicate rows:           0
Missing CUSTOMER_ID:      0
Missing timestamps:       0
```

This produces:

```text
data/interim/sgsc_selected_households.csv
```

which is used as the input to the preprocessing pipeline.

---

## 5. SGSC Energy Preprocessing

The extracted interval dataset contains:

```text
1,247,602 interval readings
30 households
```

During preprocessing:

```text
Invalid timestamps:          0
Duplicate readings removed:  118
```

For each household, the pipeline reconstructs the expected continuous 30-minute time index.

Missing intervals are filled using time-aware interpolation.

A rolling 3-sigma method is then used for outlier treatment:

```text
Rolling window: 48 readings
Window duration: 24 hours
Threshold:       mean ± 3 standard deviations
```

Detected outliers are temporarily replaced with missing values and reconstructed using time-aware interpolation.

### Hourly Resampling

SGSC energy fields represent energy consumed or generated during each 30-minute interval.

Therefore, two 30-minute readings are aggregated to hourly energy using:

```text
SUM
```

rather than mean.

The final processed dataset contains:

```text
625,424 hourly readings
30 households
```

Final validation:

```text
Missing GENERAL_SUPPLY_KWH:             0
Duplicate household/timestamp pairs:    0
Negative GENERAL_SUPPLY_KWH:            0
Non-hour-aligned timestamps:            0
Missing hourly timestamps:              0
```

Hourly continuity was checked independently for every household.

All 30 households have:

```text
missing_hours = 0
```

---

## 6. Hourly Energy Rows by Household

| Household ID | Hourly Rows |
|---|---:|
| 8162493 | 20,946 |
| 8163939 | 20,735 |
| 10042604 | 20,950 |
| 10042756 | 20,736 |
| 10042934 | 20,800 |
| 10042974 | 20,999 |
| 10174306 | 20,996 |
| 10257896 | 20,254 |
| 10305996 | 20,972 |
| 10341486 | 20,978 |
| 10346126 | 20,950 |
| 10346582 | 20,954 |
| 10390745 | 20,975 |
| 10413359 | 20,999 |
| 10508173 | 20,923 |
| 10508205 | 20,716 |
| 10534327 | 21,002 |
| 10534355 | 21,004 |
| 10673547 | 20,288 |
| 10673789 | 20,928 |
| 10733766 | 20,730 |
| 10733884 | 20,978 |
| 10760980 | 20,949 |
| 10850536 | 20,828 |
| 10859314 | 20,879 |
| 10883534 | 20,576 |
| 10885220 | 20,951 |
| 11006864 | 20,904 |
| 11228008 | 20,787 |
| 11446117 | 20,737 |
| **Total** | **625,424** |

---

## 7. Loading SGSC Data into PostgreSQL

`ingest_csv.py` loads both household metadata and cleaned hourly SGSC energy data into PostgreSQL.

Household metadata is mapped to the existing database schema as follows:

| SGSC Field | Database Column |
|---|---|
| `CUSTOMER_ID` | `household_id` |
| `DWELLING_TYPE_CD` | `dwelling_type` |
| `ASSRTD_CLIMATE_ZONE_DESC` | `climate_zone` |
| `ASSRTD_ELECTRICITY_USE_GRP_CD` | `electricity_use_group` |
| `HAS_AIRCON` | `has_aircon` |
| `HAS_GENERATION` | `has_generation` |

The ingestion process:

- validates required energy columns;
- validates timestamps;
- checks duplicate household/timestamp records;
- checks negative consumption values;
- loads household metadata;
- inserts households before energy data to satisfy foreign-key requirements;
- loads energy readings in batches of 5,000;
- uses conflict handling to safely update existing records; and
- verifies database counts after ingestion.

Run inside Docker:

```bash
docker compose exec backend python data_pipeline/ingest_csv.py
```

Final verification:

```text
SGSC households:               30
SGSC energy readings:          625,424
Households with core metadata: 30

SGSC ingestion completed successfully.
```

---

## 8. Open-Meteo Weather Integration

`weather.py` retrieves historical hourly weather corresponding to the SGSC energy period.

The weather data currently covers:

```text
2011-10-10 07:00:00
to
2014-03-03 10:00:00
```

The weather pipeline uses the SGSC dataset-level geographic reference:

```text
Latitude:  -32.92921
Longitude: 151.77002
Timezone:  Australia/Sydney
```

This is a common dataset-level reference location.

Individual household coordinates are not available in the current dataset, so household-specific coordinates are not fabricated.

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

`shortwave_radiation` provides the solar-radiation weather feature required by the project.

The common weather series is associated with each selected SGSC household to provide a direct household/timestamp join interface for downstream processing.

Run:

```bash
docker compose exec backend python data_pipeline/weather.py
```

Final database result:

```text
Weather rows: 630,120
Households:   30

Start: 2011-10-10 07:00:00
End:   2014-03-03 10:00:00

Open-Meteo ingestion completed successfully.
```

There are:

```text
21,004 weather rows per household
```

for a total of:

```text
21,004 × 30 = 630,120 weather rows
```

---

## 9. Energy + Weather Integration

Energy and weather records are joined using:

```text
household_id
+
reading_timestamp
```

Validation was performed using a left join from SGSC energy readings to Open-Meteo weather readings.

Final result:

| Metric | Records |
|---|---:|
| Total SGSC energy rows | 625,424 |
| Matched weather rows | 625,424 |
| Missing weather rows | 0 |
| Weather coverage | **100.00%** |

Therefore, every SGSC hourly energy reading in the current dataset has corresponding historical weather information.

---

# Database Setup

## 10. PostgreSQL with Docker

The development database runs through Docker Compose.

Current database service:

```text
PostgreSQL 16
Database: smart_energy
User: secpps
Container: secpps_db
```

The current implementation uses standard PostgreSQL 16 and does **not** depend on TimescaleDB-specific functionality.

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

The database service should report a healthy status.

---

## 11. Backend Database Connection

Inside Docker, the backend connects to PostgreSQL using:

```text
postgresql://secpps:secpps_dev_password@db:5432/smart_energy
```

This is configured through the `DATABASE_URL` environment variable.

Inside Docker:

```text
db:5432
```

is the database address.

Do not replace `db` with `localhost` when code is running inside the backend container.

From the host machine, PostgreSQL is exposed on:

```text
localhost:5432
```

provided another local PostgreSQL service is not already using that port.

---

## 12. Connecting to PostgreSQL

Open PostgreSQL inside the database container:

```bash
docker compose exec db psql -U secpps -d smart_energy
```

Useful commands:

```sql
\dt
```

Lists tables.

```sql
\d households
```

Shows the household schema.

```sql
\d energy_readings
```

Shows the energy schema.

```sql
\d weather_readings
```

Shows the weather schema.

Exit with:

```sql
\q
```

---

## 13. Checking Database Contents

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

Expected values:

```text
households:
SGSC          30

energy_readings:
SGSC          625424

weather_readings:
Open-Meteo    630120
```

---

# Database Dump

## 14. `smart_energy.dump`

A reproducible PostgreSQL database dump is included at:

```text
infra/init-db/smart_energy.dump
```

The current verified dump contains:

```text
30 SGSC households
625,424 SGSC hourly energy readings
630,120 Open-Meteo weather readings
```

It also contains the PostgreSQL table structures, constraints, indexes, sequences, household metadata, and weather data.

HomeC is not part of the current database snapshot.

---

## 15. Automatic Dump Restore

The repository contains:

```text
infra/init-db/02-restore.sh
```

The PostgreSQL Docker container executes initialization scripts from:

```text
/docker-entrypoint-initdb.d/
```

The restore script restores:

```text
smart_energy.dump
```

when PostgreSQL initializes a **new empty Docker volume**.

PostgreSQL initialization scripts run only when the database volume is first created.

They do not run every time the container restarts.

Therefore:

```bash
docker compose stop
docker compose start
```

does not restore the dump again.

---

## 16. Manually Restoring the Dump

Create a temporary empty database:

```bash
docker compose exec db createdb \
  -U secpps \
  smart_energy_restore
```

Restore:

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

## 17. Creating an Updated Dump

After intentionally changing and validating the database:

```bash
docker compose exec -T db pg_dump \
  -U secpps \
  -d smart_energy \
  -Fc \
  --no-owner \
  > infra/init-db/smart_energy.dump
```

Do not regenerate the shared dump from an unverified development database.

---

## 18. Verifying the Dump

For strong verification, restore the dump into a temporary database:

```bash
docker compose exec db createdb \
  -U secpps \
  smart_energy_test
```

Restore it:

```bash
docker compose exec -T db pg_restore \
  -U secpps \
  -d smart_energy_test \
  --no-owner \
  < infra/init-db/smart_energy.dump
```

Verify the important record counts:

```bash
docker compose exec db psql \
  -U secpps \
  -d smart_energy_test \
  -c "
SELECT
    (SELECT COUNT(*)
     FROM households
     WHERE source = 'SGSC') AS households,

    (SELECT COUNT(*)
     FROM energy_readings
     WHERE source = 'SGSC') AS energy_rows,

    (SELECT COUNT(*)
     FROM weather_readings
     WHERE source = 'Open-Meteo') AS weather_rows;
"
```

Expected:

```text
households | energy_rows | weather_rows
-----------+-------------+-------------
30         | 625424      | 630120
```

The current dump has been restore-tested successfully against these counts.

After verification:

```bash
docker compose exec db dropdb \
  -U secpps \
  smart_energy_test
```

---

# Running the Data Pipeline

## 19. Pipeline Order

When rebuilding from source rather than using the populated database dump, run the pipeline in the following order.

### Step 1 — Select and Extract SGSC Households

The selection stage can be run independently to generate the 30-household selection.

Extraction from the full SGSC interval file is performed with:

```bash
python backend/data_pipeline/extract_sgsc.py --extract
```

This produces:

```text
data/interim/sgsc_selected_30_households.csv
data/interim/sgsc_selected_households.csv
```

The full source SGSC interval file is intentionally not stored in the repository because of its size.

### Step 2 — Clean SGSC

From the repository root:

```bash
python backend/data_pipeline/clean.py
```

Produces:

```text
data/processed/sgsc_hourly_clean.csv
```

Expected:

```text
625,424 hourly readings
30 households
0 missing hourly timestamps
```

### Step 3 — Load SGSC into PostgreSQL

Run inside Docker:

```bash
docker compose exec backend python data_pipeline/ingest_csv.py
```

Expected:

```text
SGSC households:               30
SGSC energy readings:          625,424
Households with core metadata: 30
```

### Step 4 — Retrieve and Load Weather

```bash
docker compose exec backend python data_pipeline/weather.py
```

Expected:

```text
Rows:       630,120
Households: 30
```

### Step 5 — Validate Energy/Weather Coverage

```sql
SELECT
    COUNT(*) AS total_energy_rows,
    COUNT(w.id) AS matched_weather_rows,
    COUNT(*) - COUNT(w.id) AS missing_weather_rows,
    ROUND(
        100.0 * COUNT(w.id) / COUNT(*),
        2
    ) AS weather_coverage_percent
FROM energy_readings e
LEFT JOIN weather_readings w
    ON e.household_id = w.household_id
   AND e.reading_timestamp = w.reading_timestamp
   AND w.source = 'Open-Meteo'
WHERE e.source = 'SGSC';
```

Expected:

```text
total_energy_rows:        625424
matched_weather_rows:     625424
missing_weather_rows:     0
weather_coverage_percent: 100.00
```

---

# Downstream Usage

## 20. Data Available to ML and Forecasting

The data engineering pipeline provides:

- continuous hourly household energy readings;
- household metadata;
- aligned historical weather data; and
- a validated household/timestamp interface for downstream ML components.

A downstream query can join the datasets with:

```sql
SELECT
    e.household_id,
    e.reading_timestamp,
    e.consumption_kwh,
    e.controlled_load_kwh,
    e.gross_generation_kwh,
    e.net_generation_kwh,
    e.other_kwh,
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
ORDER BY
    e.household_id,
    e.reading_timestamp;
```

This produces the integrated hourly dataset required for downstream feature engineering, anomaly detection, forecasting, and related modelling.

Feature engineering and model implementation are handled by the corresponding ML/AI components rather than inside the data-engineering pipeline.

---

## 21. Important Notes

- The current Docker database uses PostgreSQL 16.
- The current implementation does not use TimescaleDB-specific functionality.
- The database schema has not been modified for the 30-household expansion.
- SGSC household IDs are stored using their original numeric IDs.
- Energy readings are stored at hourly resolution.
- SGSC 30-minute kWh readings are aggregated to hourly kWh using `SUM`.
- The database currently uses `TIMESTAMP WITHOUT TIME ZONE`.
- Open-Meteo is requested using `Australia/Sydney`.
- The Open-Meteo location is a common SGSC dataset-level reference point rather than an individual household location.
- All 625,424 SGSC energy readings have corresponding Open-Meteo weather records.
- The selected 30-household dataset does not contain generation households because no generation households met the final reading-count and energy-quality selection criteria.
- HomeC is not part of the active data pipeline.
- Raw, intermediate, and processed datasets should not be committed to Git unless explicitly required.
- The full multi-GB SGSC interval source file should not be committed to the repository.
- `smart_energy.dump` is the reproducible populated database snapshot for the team.
- The current database dump has been independently restored and verified with 30 households, 625,424 energy readings, and 630,120 weather readings.