import os
from pathlib import Path

import pandas as pd
import requests
from sqlalchemy import create_engine, text


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://secpps:secpps_dev_password@db:5432/smart_energy",
)

# Official SGSC dataset-level reference point
LATITUDE = -32.92921
LONGITUDE = 151.77002

TIMEZONE = "Australia/Sydney"
SOURCE = "Open-Meteo"

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
]


def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def get_sgsc_date_range(engine):
    query = text("""
        SELECT
            MIN(reading_timestamp) AS start_time,
            MAX(reading_timestamp) AS end_time
        FROM energy_readings
        WHERE source = 'SGSC';
    """)

    with engine.connect() as connection:
        row = connection.execute(query).mappings().one()

    if row["start_time"] is None or row["end_time"] is None:
        raise RuntimeError("No SGSC energy readings found in the database.")

    return row["start_time"], row["end_time"]


def get_sgsc_households(engine):
    query = text("""
        SELECT household_id
        FROM households
        WHERE source = 'SGSC'
        ORDER BY household_id;
    """)

    with engine.connect() as connection:
        rows = connection.execute(query).fetchall()

    household_ids = [row[0] for row in rows]

    if not household_ids:
        raise RuntimeError("No SGSC households found in the database.")

    return household_ids


def fetch_weather(start_time, end_time):
    start_date = pd.Timestamp(start_time).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(end_time).strftime("%Y-%m-%d")

    print("Fetching Open-Meteo historical weather")
    print(f"Location: {LATITUDE}, {LONGITUDE}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Timezone: {TIMEZONE}")

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": TIMEZONE,
    }

    response = requests.get(
        OPEN_METEO_URL,
        params=params,
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise RuntimeError(
            f"Open-Meteo response did not contain hourly data: {data}"
        )

    df = pd.DataFrame(data["hourly"])

    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    if df["time"].isna().any():
        raise RuntimeError("Invalid timestamps found in Open-Meteo response.")

    return df


def prepare_weather_data(df, start_time, end_time):
    column_map = {
        "time": "reading_timestamp",
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "dew_point_2m": "dew_point",
        "apparent_temperature": "apparent_temperature",
        "precipitation": "precip_intensity",
        "pressure_msl": "pressure",
        "cloud_cover": "cloud_cover",
        "wind_speed_10m": "wind_speed",
        "wind_direction_10m": "wind_bearing",
        "shortwave_radiation": "shortwave_radiation",
    }

    df = df.rename(columns=column_map)

    # Keep only timestamps actually covered by SGSC.
    df = df[
        (df["reading_timestamp"] >= pd.Timestamp(start_time))
        & (df["reading_timestamp"] <= pd.Timestamp(end_time))
    ].copy()

    numeric_columns = [
        "temperature",
        "humidity",
        "dew_point",
        "apparent_temperature",
        "precip_intensity",
        "pressure",
        "cloud_cover",
        "wind_speed",
        "wind_bearing",
        "shortwave_radiation",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.drop_duplicates(subset=["reading_timestamp"])
    df = df.sort_values("reading_timestamp").reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No weather rows remain after SGSC date filtering.")

    print(f"Weather rows prepared: {len(df):,}")

    return df


def clear_existing_open_meteo(engine):
    with engine.begin() as connection:
        result = connection.execute(
            text("""
                DELETE FROM weather_readings
                WHERE source = :source;
            """),
            {"source": SOURCE},
        )

    print(f"Removed existing Open-Meteo rows: {result.rowcount:,}")


def insert_weather(engine, weather_df, household_ids):
    insert_sql = text("""
        INSERT INTO weather_readings (
            household_id,
            reading_timestamp,
            temperature,
            humidity,
            apparent_temperature,
            pressure,
            wind_speed,
            cloud_cover,
            wind_bearing,
            precip_intensity,
            dew_point,
            shortwave_radiation,
            source
        )
        VALUES (
            :household_id,
            :reading_timestamp,
            :temperature,
            :humidity,
            :apparent_temperature,
            :pressure,
            :wind_speed,
            :cloud_cover,
            :wind_bearing,
            :precip_intensity,
            :dew_point,
            :shortwave_radiation,
            :source
        );
    """)

    total_inserted = 0

    for household_id in household_ids:
        rows = []

        for record in weather_df.to_dict("records"):
            rows.append(
                {
                    "household_id": household_id,
                    "reading_timestamp": record["reading_timestamp"],
                    "temperature": record["temperature"],
                    "humidity": record["humidity"],
                    "apparent_temperature": record["apparent_temperature"],
                    "pressure": record["pressure"],
                    "wind_speed": record["wind_speed"],
                    "cloud_cover": record["cloud_cover"],
                    "wind_bearing": record["wind_bearing"],
                    "precip_intensity": record["precip_intensity"],
                    "dew_point": record["dew_point"],
                    "shortwave_radiation": record["shortwave_radiation"],
                    "source": SOURCE,
                }
            )

        with engine.begin() as connection:
            connection.execute(insert_sql, rows)

        total_inserted += len(rows)

        print(
            f"Inserted household {household_id}: "
            f"{len(rows):,} weather rows"
        )

    print(f"Total Open-Meteo rows inserted: {total_inserted:,}")


def verify_weather(engine):
    query = text("""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT household_id) AS households,
            MIN(reading_timestamp) AS start_time,
            MAX(reading_timestamp) AS end_time
        FROM weather_readings
        WHERE source = 'Open-Meteo';
    """)

    with engine.connect() as connection:
        row = connection.execute(query).mappings().one()

    print("\nWeather database verification")
    print("-----------------------------")
    print(f"Rows:       {row['total_rows']:,}")
    print(f"Households: {row['households']}")
    print(f"Start:      {row['start_time']}")
    print(f"End:        {row['end_time']}")


def main():
    engine = get_engine()

    start_time, end_time = get_sgsc_date_range(engine)
    household_ids = get_sgsc_households(engine)

    print(f"SGSC households: {len(household_ids)}")
    print(f"SGSC start: {start_time}")
    print(f"SGSC end:   {end_time}")

    weather_df = fetch_weather(start_time, end_time)
    weather_df = prepare_weather_data(
        weather_df,
        start_time,
        end_time,
    )

    clear_existing_open_meteo(engine)
    insert_weather(
        engine,
        weather_df,
        household_ids,
    )

    verify_weather(engine)

    print("\nOpen-Meteo ingestion completed successfully.")


if __name__ == "__main__":
    main()