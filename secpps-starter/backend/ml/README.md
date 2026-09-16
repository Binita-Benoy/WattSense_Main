# WattSense — Anomaly Detection Project

The project is separated into two independently runnable stages:

- `feature_engineering/` — US-10 feature creation
- `anomaly_detection/` — US-11 to US-14 detection, evaluation and API output
- root-level files — shared configuration, database loading, splitting and runners

## Project structure

```text
anomaly/
├── config.py
├── db_loader.py
├── data_utils.py
├── splitter.py
├── run_feature_engineering.py
├── run_anomaly_detection.py
│
├── feature_engineering/
│   ├── __init__.py
│   └── us10_features.py
│
├── anomaly_detection/
│   ├── __init__.py
│   ├── us11_isolation_forest.py
│   ├── us12_autoencoder.py
│   ├── us13_evaluation.py
│   └── us14_api.py
│
└── outputs/
    ├── features/
    ├── models/
    ├── results/
    ├── api/
    └── graphs/
```

## How to run

Run these commands from inside the `anomaly` project folder.

### Stage 1 — Feature engineering

```bash
python run_feature_engineering.py
```

This loads the PostgreSQL data, creates the US-10 feature matrix and saves:

`outputs/features/feature_matrix.csv`

### Stage 2 — Anomaly detection

Run this after feature engineering:

```bash
python run_anomaly_detection.py
```

This performs the chronological train/validation/test split and runs:

- US-11 Isolation Forest
- US-12 LSTM Autoencoder
- US-13 injection-based evaluation
- US-14 API output generation

The anomaly stage reuses the saved feature matrix instead of rebuilding it.

## Database

Configure the PostgreSQL connection in `.env`. The default database name is
`smart_energy`.

Example restore commands:

```bash
createdb -U postgres smart_energy
pg_restore -U postgres -d smart_energy data/smart_energy.dump
```

## Notes

The previous `run_all.py` master runner has been removed. The older duplicate
`prepare_anomaly_data.py` and `isolation_forest.py` implementations have also
been removed because their functionality is superseded by US-10 and US-11.

## Updated anomaly pipeline

### Weather features
US-10 now joins `weather_readings` to `energy_readings` by `household_id` and `reading_timestamp`. It uses temperature, apparent temperature, humidity, pressure, wind speed, cloud cover, precipitation, dew point, precipitation probability and shortwave radiation, plus derived weather-demand features such as temperature deviation, heating/cooling degree indicators and 24-hour weather baselines.

### Anomaly detection features
US-11 Isolation Forest now uses a broader set of energy, temporal, calendar and weather features. US-12 uses a selected multivariate subset in the LSTM autoencoder, so the sequence is no longer just `24 x 1` kWh; it is `24 x N` where N is the number of available AE features.

### Logging
The two main runners no longer show the long pipeline output in the terminal. Output from Python `print()` statements and TensorFlow is captured into timestamped files under:

`outputs/results/logs/`

Run:

```text
python run_feature_engineering.py
python run_anomaly_detection.py
```

Check the newest `.log` file under `outputs/results/logs/` for the complete run.
