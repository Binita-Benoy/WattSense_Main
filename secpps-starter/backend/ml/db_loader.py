"""Database loader for the WattSense ML pipeline."""
from __future__ import annotations

import logging
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from config import (
    HOUSEHOLD_IDS,
    DB_TIMESTAMP,
    DB_KWH,
    COL_KWH,
    COL_TIMESTAMP,
    COL_HOUSEHOLD,
    COL_PUBLIC_HOLIDAY,
)
from logging_utils import log_print

logger = logging.getLogger()

load_dotenv(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
)


def get_engine():
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "smart_energy")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")
    return create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}",
        pool_pre_ping=True,
    )


def _hh_sql() -> str:
    return ",".join(str(int(h)) for h in HOUSEHOLD_IDS)


def load_energy_readings() -> pd.DataFrame:
    """Load hourly energy readings, including the public-holiday flag."""
    engine = get_engine()
    query = f"""
        SELECT household_id,
               reading_timestamp,
               consumption_kwh,
               controlled_load_kwh,
               gross_generation_kwh,
               net_generation_kwh,
               other_kwh,
               source,
               is_public_holiday
        FROM energy_readings
        WHERE household_id IN ({_hh_sql()})
        ORDER BY household_id, reading_timestamp
    """
    df = pd.read_sql(query, engine, parse_dates=[DB_TIMESTAMP])
    df = df.rename(columns={DB_TIMESTAMP: COL_TIMESTAMP, DB_KWH: COL_KWH})

    # PostgreSQL boolean -> numeric model feature.
    if COL_PUBLIC_HOLIDAY in df.columns:
        df[COL_PUBLIC_HOLIDAY] = (
            df[COL_PUBLIC_HOLIDAY].fillna(False).astype(bool).astype("int8")
        )

    df = df.set_index(COL_TIMESTAMP)
    df.index.name = COL_TIMESTAMP
    return df


def load_weather_readings() -> pd.DataFrame:
    """Load numeric weather covariates aligned to household/timestamp."""
    engine = get_engine()
    query = f"""
        SELECT household_id,
               reading_timestamp,
               temperature,
               humidity,
               visibility,
               apparent_temperature,
               pressure,
               wind_speed,
               cloud_cover,
               wind_bearing,
               precip_intensity,
               dew_point,
               precip_probability,
               shortwave_radiation
        FROM weather_readings
        WHERE household_id IN ({_hh_sql()})
        ORDER BY household_id, reading_timestamp
    """
    df = pd.read_sql(query, engine, parse_dates=[DB_TIMESTAMP])
    return df.rename(columns={DB_TIMESTAMP: COL_TIMESTAMP})


def load_all_households(verbose: bool = True) -> pd.DataFrame:
    """Load energy + weather and merge them on household and timestamp."""
    energy = load_energy_readings().reset_index()
    weather = load_weather_readings()

    weather_cols = [
        "temperature",
        "humidity",
        "visibility",
        "apparent_temperature",
        "pressure",
        "wind_speed",
        "cloud_cover",
        "wind_bearing",
        "precip_intensity",
        "dew_point",
        "precip_probability",
        "shortwave_radiation",
    ]

    merged = energy.merge(
        weather,
        on=[COL_HOUSEHOLD, COL_TIMESTAMP],
        how="left",
        suffixes=("", "_weather"),
        validate="one_to_one",
    )
    merged = merged.set_index(COL_TIMESTAMP).sort_index()

    for col in weather_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    if verbose:
        log_print("ALL HOUSEHOLDS LOADED WITH WEATHER + HOLIDAY CONTEXT")
        log_print(f"Total rows: {len(merged):,}")
        log_print(f"Households: {merged[COL_HOUSEHOLD].nunique()}")
        if COL_PUBLIC_HOLIDAY in merged.columns:
            log_print(
                f"Public-holiday rows: {int(merged[COL_PUBLIC_HOLIDAY].sum()):,}"
            )
        available_weather = [c for c in weather_cols if c in merged.columns]
        if available_weather:
            missing = merged[available_weather].isna().mean().sort_values(ascending=False)
            log_print("Weather missingness (%):")
            log_print((missing * 100).round(2).to_string())
        summary = merged.groupby(COL_HOUSEHOLD)[COL_KWH].agg(
            count="count", mean="mean", max="max", min="min"
        ).round(4)
        log_print(summary.to_string())

    return merged



def load_inference_context(
    household_ids,
    start,
    end,
) -> pd.DataFrame:
    """Load raw energy + weather context for online anomaly inference.

    Unlike ``load_all_households`` this function is request-scoped: it loads
    only the requested household IDs and time range. Household IDs are coerced
    to integers before being inserted into the SQL IN clause; timestamps remain
    bound SQL parameters.
    """
    ids = sorted({int(h) for h in household_ids})
    if not ids:
        return pd.DataFrame()

    start = pd.Timestamp(start).to_pydatetime()
    end = pd.Timestamp(end).to_pydatetime()
    hh_sql = ",".join(str(h) for h in ids)
    engine = get_engine()

    energy_query = f"""
        SELECT household_id,
               reading_timestamp,
               consumption_kwh,
               controlled_load_kwh,
               gross_generation_kwh,
               net_generation_kwh,
               other_kwh,
               source,
               is_public_holiday
        FROM energy_readings
        WHERE household_id IN ({hh_sql})
          AND reading_timestamp BETWEEN %(start)s AND %(end)s
        ORDER BY household_id, reading_timestamp
    """
    weather_query = f"""
        SELECT household_id,
               reading_timestamp,
               temperature,
               humidity,
               visibility,
               apparent_temperature,
               pressure,
               wind_speed,
               cloud_cover,
               wind_bearing,
               precip_intensity,
               dew_point,
               precip_probability,
               shortwave_radiation
        FROM weather_readings
        WHERE household_id IN ({hh_sql})
          AND reading_timestamp BETWEEN %(start)s AND %(end)s
        ORDER BY household_id, reading_timestamp
    """
    params = {"start": start, "end": end}
    energy = pd.read_sql(
        energy_query,
        engine,
        params=params,
        parse_dates=[DB_TIMESTAMP],
    ).rename(columns={DB_TIMESTAMP: COL_TIMESTAMP, DB_KWH: COL_KWH})

    weather = pd.read_sql(
        weather_query,
        engine,
        params=params,
        parse_dates=[DB_TIMESTAMP],
    ).rename(columns={DB_TIMESTAMP: COL_TIMESTAMP})

    if energy.empty:
        # Incoming readings may be newer than the DB. Returning an empty/raw
        # context is preferable to fabricating historical data; US-14 will
        # report rows as unscored if enough context cannot be built.
        return pd.DataFrame()

    if COL_PUBLIC_HOLIDAY in energy.columns:
        energy[COL_PUBLIC_HOLIDAY] = (
            energy[COL_PUBLIC_HOLIDAY]
            .fillna(False)
            .astype(bool)
            .astype("int8")
        )

    if weather.empty:
        merged = energy
    else:
        merged = energy.merge(
            weather,
            on=[COL_HOUSEHOLD, COL_TIMESTAMP],
            how="left",
            validate="one_to_one",
        )

    merged = (
        merged
        .sort_values([COL_HOUSEHOLD, COL_TIMESTAMP])
        .set_index(COL_TIMESTAMP)
    )
    merged.index.name = COL_TIMESTAMP
    return merged


def list_inference_household_ids(source: str | None = None) -> list[int]:
    """Return household IDs that currently have energy readings.

    Used by US-14 when the backend requests inference for every household
    currently present in ``energy_readings``.  ``source`` can optionally
    restrict the list (for example ``source="synthetic"``).
    """
    engine = get_engine()
    if source is None:
        query = """
            SELECT DISTINCT household_id
            FROM energy_readings
            ORDER BY household_id
        """
        df = pd.read_sql(query, engine)
    else:
        query = """
            SELECT DISTINCT household_id
            FROM energy_readings
            WHERE source = %(source)s
            ORDER BY household_id
        """
        df = pd.read_sql(query, engine, params={"source": source})

    if df.empty:
        return []
    return [int(v) for v in df[COL_HOUSEHOLD].tolist()]


def get_inference_household_ranges(
    household_ids,
) -> pd.DataFrame:
    """Return min/max timestamps and row counts for requested households."""
    ids = sorted({int(h) for h in household_ids})
    if not ids:
        return pd.DataFrame(
            columns=[COL_HOUSEHOLD, "min_timestamp", "max_timestamp", "reading_count"]
        )

    hh_sql = ",".join(str(h) for h in ids)
    query = f"""
        SELECT household_id,
               MIN(reading_timestamp) AS min_timestamp,
               MAX(reading_timestamp) AS max_timestamp,
               COUNT(*) AS reading_count
        FROM energy_readings
        WHERE household_id IN ({hh_sql})
        GROUP BY household_id
        ORDER BY household_id
    """
    return pd.read_sql(
        query,
        get_engine(),
        parse_dates=["min_timestamp", "max_timestamp"],
    )


def load_inference_household_histories(household_ids) -> pd.DataFrame:
    """Load complete energy + weather histories for arbitrary households.

    This is the US-14 database entry point.  The households do not need to be
    listed in ``config.HOUSEHOLD_IDS``.  Weather is assumed to have already
    been ensured (US-14 calls the teammate's ``ensure_historical_weather``
    first), then this function reads the complete available range.
    """
    ids = sorted({int(h) for h in household_ids})
    if not ids:
        return pd.DataFrame()

    ranges = get_inference_household_ranges(ids)
    if ranges.empty:
        return pd.DataFrame()

    found = set(int(v) for v in ranges[COL_HOUSEHOLD].tolist())
    missing = [h for h in ids if h not in found]
    if missing:
        raise ValueError(
            "No energy_readings found for household(s): "
            + ", ".join(str(h) for h in missing)
        )

    return load_inference_context(
        ids,
        start=ranges["min_timestamp"].min(),
        end=ranges["max_timestamp"].max(),
    )

def load_household_meta() -> pd.DataFrame:
    engine = get_engine()
    query = f"""
        SELECT household_id,
               dwelling_type,
               climate_zone,
               electricity_use_group,
               has_aircon,
               has_generation
        FROM households
        WHERE household_id IN ({_hh_sql()})
    """
    return pd.read_sql(query, engine)


def load_inference_auxiliary_context(
    household_ids,
    start,
    end,
) -> pd.DataFrame:
    """Load only weather + public-holiday context for arbitrary households.

    This is the preferred US-14 helper when the frontend already supplies the
    complete energy history.  Consumption is never read here, so request data
    remains authoritative.  New synthetic household IDs do not need to appear
    in ``config.HOUSEHOLD_IDS``.
    """
    ids = sorted({int(h) for h in household_ids})
    if not ids:
        return pd.DataFrame()

    start = pd.Timestamp(start).to_pydatetime()
    end = pd.Timestamp(end).to_pydatetime()
    hh_sql = ",".join(str(h) for h in ids)
    engine = get_engine()
    params = {"start": start, "end": end}

    holiday_query = f"""
        SELECT household_id,
               reading_timestamp,
               is_public_holiday
        FROM energy_readings
        WHERE household_id IN ({hh_sql})
          AND reading_timestamp BETWEEN %(start)s AND %(end)s
        ORDER BY household_id, reading_timestamp
    """
    weather_query = f"""
        SELECT household_id,
               reading_timestamp,
               temperature,
               humidity,
               visibility,
               apparent_temperature,
               pressure,
               wind_speed,
               cloud_cover,
               wind_bearing,
               precip_intensity,
               dew_point,
               precip_probability,
               shortwave_radiation
        FROM weather_readings
        WHERE household_id IN ({hh_sql})
          AND reading_timestamp BETWEEN %(start)s AND %(end)s
        ORDER BY household_id, reading_timestamp
    """

    holidays = pd.read_sql(
        holiday_query,
        engine,
        params=params,
        parse_dates=[DB_TIMESTAMP],
    ).rename(columns={DB_TIMESTAMP: COL_TIMESTAMP})

    weather = pd.read_sql(
        weather_query,
        engine,
        params=params,
        parse_dates=[DB_TIMESTAMP],
    ).rename(columns={DB_TIMESTAMP: COL_TIMESTAMP})

    key = [COL_HOUSEHOLD, COL_TIMESTAMP]
    if holidays.empty and weather.empty:
        return pd.DataFrame()
    if holidays.empty:
        merged = weather
    elif weather.empty:
        merged = holidays
    else:
        merged = holidays.merge(weather, on=key, how="outer", validate="one_to_one")

    if COL_PUBLIC_HOLIDAY in merged.columns:
        merged[COL_PUBLIC_HOLIDAY] = (
            merged[COL_PUBLIC_HOLIDAY]
            .fillna(False)
            .astype(bool)
            .astype("int8")
        )

    return (
        merged
        .sort_values(key)
        .set_index(COL_TIMESTAMP)
    )


if __name__ == "__main__":
    log_print(load_all_households().head())
