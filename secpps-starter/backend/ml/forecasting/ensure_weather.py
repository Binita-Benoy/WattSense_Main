"""
Given ONLY consumption data for a household, automatically ensures matching
historical weather exists -- fetching it from Open-Meteo if it doesn't.
This is the piece that makes "just give me consumption" actually work,
instead of requiring a pre-joined dataset.

LIMITATION, stated plainly: real household location isn't in the schema
(households has climate_zone, dwelling_type, etc. -- no lat/long). For
SGSC households, we assume the Newcastle/NSW region, since that's where
the real trial was concentrated -- an approximation, not exact per-house
coordinates. For a fully general system, Alka would need to add real
lat/long to the households table.

Usage:
    from ensure_weather import ensure_historical_weather
    ensure_historical_weather(engine, household_id=10042974)
    # weather_readings now has matching rows -- safe to call the
    # existing generate_weather_forecast.py etc. right after this
"""
import os

import pandas as pd
import requests
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)

# Approximate SGSC trial region (Newcastle, NSW) -- see limitation note above
DEFAULT_LATITUDE = -32.9283
DEFAULT_LONGITUDE = 151.7817


def fetch_historical_weather(latitude, longitude, start_date, end_date) -> pd.DataFrame:
    """
    NOTE: written against Open-Meteo's documented archive API but not
    tested end-to-end by me -- verify the response shape in your own
    environment on first use.
    """
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,relative_humidity_2m,shortwave_radiation"
        "&timezone=Australia%2FSydney"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()["hourly"]

    return pd.DataFrame({
        "reading_timestamp": pd.to_datetime(data["time"]),
        "temperature": data["temperature_2m"],
        "humidity": data["relative_humidity_2m"],
        "shortwave_radiation": data["shortwave_radiation"],
    })


def ensure_historical_weather(engine, household_id: int, latitude=DEFAULT_LATITUDE, longitude=DEFAULT_LONGITUDE):
    """
    Checks if this household already has weather data covering its
    consumption date range. If not, fetches and inserts it automatically.
    Safe to call every time before forecasting -- does nothing if
    weather already exists.
    """
    with engine.connect() as conn:
        consumption_range = conn.execute(
            text("SELECT min(reading_timestamp), max(reading_timestamp) FROM energy_readings WHERE household_id = :id"),
            {"id": household_id},
        ).fetchone()

        if consumption_range[0] is None:
            raise ValueError(f"No energy_readings found for household {household_id} -- add consumption data first")

        min_ts, max_ts = consumption_range

        weather_count = conn.execute(
            text("""SELECT count(*) FROM weather_readings
                     WHERE household_id = :id
                     AND reading_timestamp BETWEEN :min_ts AND :max_ts"""),
            {"id": household_id, "min_ts": min_ts, "max_ts": max_ts},
        ).scalar()

        expected_hours = int((max_ts - min_ts).total_seconds() / 3600)

    if weather_count >= expected_hours * 0.9:  # allow small gaps
        print(f"Household {household_id}: weather already covers its consumption range ({weather_count} rows) -- nothing to do")
        return

    print(f"Household {household_id}: weather missing or incomplete ({weather_count}/{expected_hours} expected) -- fetching from Open-Meteo...")

    weather_df = fetch_historical_weather(
        latitude, longitude, min_ts.date().isoformat(), max_ts.date().isoformat()
    )

    rows = [
        {
            "household_id": household_id, "reading_timestamp": r.reading_timestamp,
            "temperature": round(float(r.temperature), 2), "humidity": round(float(r.humidity), 2),
            "shortwave_radiation": round(float(r.shortwave_radiation), 2), "source": "open_meteo_auto",
        }
        for r in weather_df.itertuples()
    ]
    with engine.begin() as conn:
        conn.execute(
            text("""INSERT INTO weather_readings (household_id, reading_timestamp, temperature, humidity, shortwave_radiation, source)
                     VALUES (:household_id, :reading_timestamp, :temperature, :humidity, :shortwave_radiation, :source)
                     ON CONFLICT DO NOTHING"""),
            rows,
        )
    print(f"Household {household_id}: inserted {len(rows)} weather rows automatically (source='open_meteo_auto')")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--household-id", type=int, required=True)
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)
    ensure_historical_weather(engine, args.household_id)
