from logging_utils import log_print
import logging
logger = logging.getLogger()
# Database loader for the WattSense anomaly-detection pipeline.
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

from config import (
    HOUSEHOLD_IDS, DB_TIMESTAMP, DB_KWH, DB_HOUSEHOLD,
    COL_KWH, COL_TIMESTAMP, COL_HOUSEHOLD
)

load_dotenv(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
))


def get_engine():
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "smart_energy")
    user = os.getenv("DB_USER", "postgres")
    pw = os.getenv("DB_PASSWORD", "postgres")
    return create_engine(
        f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{name}"
    )


def _hh_sql():
    return ",".join(str(h) for h in HOUSEHOLD_IDS)


def load_energy_readings() -> pd.DataFrame:
    engine = get_engine()
    query = f"""
        SELECT household_id, reading_timestamp, consumption_kwh,
               controlled_load_kwh, gross_generation_kwh,
               net_generation_kwh, other_kwh, source
        FROM energy_readings
        WHERE household_id IN ({_hh_sql()})
        ORDER BY household_id, reading_timestamp
    """
    df = pd.read_sql(query, engine, parse_dates=[DB_TIMESTAMP])
    df = df.rename(columns={DB_TIMESTAMP: COL_TIMESTAMP, DB_KWH: COL_KWH})
    df = df.set_index(COL_TIMESTAMP)
    df.index.name = COL_TIMESTAMP
    return df


def load_weather_readings() -> pd.DataFrame:
    """Load numeric weather covariates aligned to each household/timestamp."""
    engine = get_engine()
    query = f"""
        SELECT household_id, reading_timestamp,
               temperature, humidity, visibility, apparent_temperature,
               pressure, wind_speed, cloud_cover, wind_bearing,
               precip_intensity, dew_point, precip_probability,
               shortwave_radiation
        FROM weather_readings
        WHERE household_id IN ({_hh_sql()})
        ORDER BY household_id, reading_timestamp
    """
    df = pd.read_sql(query, engine, parse_dates=[DB_TIMESTAMP])
    df = df.rename(columns={DB_TIMESTAMP: COL_TIMESTAMP})
    return df


def load_all_households(verbose: bool = True) -> pd.DataFrame:
    """Load energy and weather readings and merge on household + timestamp."""
    energy = load_energy_readings()
    weather = load_weather_readings()

    weather = weather.set_index(COL_TIMESTAMP)
    weather.index.name = COL_TIMESTAMP

    # Keep weather names explicit so they are easy to identify in the feature matrix.
    weather_cols = [
        "temperature", "humidity", "visibility", "apparent_temperature",
        "pressure", "wind_speed", "cloud_cover", "wind_bearing",
        "precip_intensity", "dew_point", "precip_probability",
        "shortwave_radiation"
    ]

    merged = energy.reset_index().merge(
        weather.reset_index(),
        on=[COL_HOUSEHOLD, COL_TIMESTAMP],
        how="left",
        suffixes=("", "_weather")
    )
    merged = merged.set_index(COL_TIMESTAMP).sort_index()

    # Numeric conversion: blank strings become NaN.
    for col in weather_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    if verbose:
        log_print("ALL HOUSEHOLDS LOADED WITH WEATHER")
        log_print(f"Total rows: {len(merged):,}")
        log_print(f"Columns: {list(merged.columns)}")
        missing = merged[weather_cols].isna().mean().sort_values(ascending=False)
        log_print("Weather missingness (%):")
        log_print((missing * 100).round(2).to_string())
        summary = merged.groupby(COL_HOUSEHOLD)[COL_KWH].agg(
            count="count", mean="mean", max="max", min="min"
        ).round(4)
        log_print(summary.to_string())

    return merged


def load_household_meta() -> pd.DataFrame:
    engine = get_engine()
    query = f"""
        SELECT household_id, dwelling_type, climate_zone,
               electricity_use_group, has_aircon, has_generation
        FROM households
        WHERE household_id IN ({_hh_sql()})
    """
    return pd.read_sql(query, engine)


if __name__ == "__main__":
    log_print(load_all_households().head())
