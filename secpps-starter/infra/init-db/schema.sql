CREATE TABLE IF NOT EXISTS households (
    household_id BIGINT PRIMARY KEY,
    source VARCHAR(50) NOT NULL,
    dwelling_type VARCHAR(100),
    climate_zone VARCHAR(100),
    electricity_use_group VARCHAR(50),
    has_aircon BOOLEAN,
    has_generation BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS energy_readings (
    id BIGSERIAL PRIMARY KEY,
    household_id BIGINT NOT NULL,
    reading_timestamp TIMESTAMP NOT NULL,
    consumption_kwh DOUBLE PRECISION,
    controlled_load_kwh DOUBLE PRECISION,
    gross_generation_kwh DOUBLE PRECISION,
    net_generation_kwh DOUBLE PRECISION,
    other_kwh DOUBLE PRECISION,
    source VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (household_id)
        REFERENCES households(household_id),
    UNIQUE (household_id, reading_timestamp)
);

CREATE TABLE IF NOT EXISTS appliance_readings (
    id BIGSERIAL PRIMARY KEY,
    household_id BIGINT NOT NULL,
    reading_timestamp TIMESTAMP NOT NULL,
    appliance_name VARCHAR(100) NOT NULL,
    power_kw DOUBLE PRECISION,
    source VARCHAR(50) NOT NULL,
    FOREIGN KEY (household_id)
        REFERENCES households(household_id)
);

CREATE TABLE IF NOT EXISTS weather_readings (
    id BIGSERIAL PRIMARY KEY,
    household_id BIGINT,
    reading_timestamp TIMESTAMP NOT NULL,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    visibility DOUBLE PRECISION,
    apparent_temperature DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION,
    cloud_cover DOUBLE PRECISION,
    wind_bearing DOUBLE PRECISION,
    precip_intensity DOUBLE PRECISION,
    dew_point DOUBLE PRECISION,
    precip_probability DOUBLE PRECISION,
    weather_icon VARCHAR(100),
    weather_summary VARCHAR(255),
    shortwave_radiation DOUBLE PRECISION,
    source VARCHAR(50) NOT NULL,
    FOREIGN KEY (household_id)
        REFERENCES households(household_id)
);

CREATE INDEX IF NOT EXISTS idx_energy_household_time
ON energy_readings (household_id, reading_timestamp);

CREATE INDEX IF NOT EXISTS idx_appliance_household_time
ON appliance_readings (household_id, reading_timestamp);

CREATE INDEX IF NOT EXISTS idx_weather_household_time
ON weather_readings (household_id, reading_timestamp);