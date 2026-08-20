-- Runs automatically the first time the db container starts (empty volume only).
-- If you change this after the volume already exists, you must:
--   docker compose down -v   (drops the db volume!) then docker compose up again.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ---------------------------------------------------------------------
-- Core tables (Alka: adjust freely, this is just a starting skeleton)
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS households (
    household_id     TEXT PRIMARY KEY,
    dwelling_type     TEXT,
    tariff_type       TEXT,
    postcode          TEXT,
    created_at        TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS energy_readings (
    household_id      TEXT NOT NULL REFERENCES households(household_id),
    ts                TIMESTAMPTZ NOT NULL,
    kwh                DOUBLE PRECISION NOT NULL,
    source             TEXT DEFAULT 'sgsc',       -- e.g. 'sgsc', 'mqtt', 'csv_upload'
    PRIMARY KEY (household_id, ts)
);

-- Turn energy_readings into a TimescaleDB hypertable, partitioned by time
SELECT create_hypertable('energy_readings', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_energy_readings_household_ts
    ON energy_readings (household_id, ts DESC);

CREATE TABLE IF NOT EXISTS weather_readings (
    postcode          TEXT NOT NULL,
    ts                TIMESTAMPTZ NOT NULL,
    temperature_c      DOUBLE PRECISION,
    humidity_pct        DOUBLE PRECISION,
    solar_irradiance   DOUBLE PRECISION,
    PRIMARY KEY (postcode, ts)
);

SELECT create_hypertable('weather_readings', 'ts', if_not_exists => TRUE);
