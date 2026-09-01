"""
SGSC cleaned-data ingestion pipeline.

Loads the cleaned, hourly SGSC dataset into the WattSense PostgreSQL /
TimescaleDB database.

Owner: Alka - Data Engineering
"""

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://secpps:secpps_dev_password@db:5432/smart_energy",
)

BASE_DIR = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_FILE = (
    BASE_DIR / "data" / "processed" / "sgsc_hourly_clean.csv"
)

BATCH_SIZE = 5000


def get_engine():
    """Create a SQLAlchemy database engine."""
    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
    )


def validate_input(df: pd.DataFrame) -> None:
    """Validate the cleaned SGSC dataset before database ingestion."""

    required_columns = {
        "CUSTOMER_ID",
        "READING_DATETIME",
        "GENERAL_SUPPLY_KWH",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df["CUSTOMER_ID"].isna().any():
        raise ValueError("CUSTOMER_ID contains missing values.")

    if df["READING_DATETIME"].isna().any():
        raise ValueError("READING_DATETIME contains missing values.")

    if df["GENERAL_SUPPLY_KWH"].isna().any():
        raise ValueError("GENERAL_SUPPLY_KWH contains missing values.")

    duplicates = df.duplicated(
        subset=["CUSTOMER_ID", "READING_DATETIME"]
    ).sum()

    if duplicates:
        raise ValueError(
            f"Found {duplicates} duplicate household/timestamp rows."
        )

    negative = (df["GENERAL_SUPPLY_KWH"] < 0).sum()

    if negative:
        raise ValueError(
            f"Found {negative} negative consumption readings."
        )


def load_households(connection, household_ids):
    """
    Ensure all SGSC households exist before energy readings are inserted.

    Household metadata can be enriched separately from the SGSC household
    metadata dataset.
    """

    statement = text(
        """
        INSERT INTO households (
            household_id,
            source
        )
        VALUES (
            :household_id,
            'SGSC'
        )
        ON CONFLICT (household_id)
        DO UPDATE SET
            source = EXCLUDED.source;
        """
    )

    rows = [
        {"household_id": int(household_id)}
        for household_id in household_ids
    ]

    connection.execute(statement, rows)


def prepare_energy_rows(df: pd.DataFrame):
    """Convert cleaned dataframe records to database records."""

    optional_columns = {
        "CONTROLLED_LOAD_KWH": "controlled_load_kwh",
        "GROSS_GENERATION_KWH": "gross_generation_kwh",
        "NET_GENERATION_KWH": "net_generation_kwh",
        "OTHER_KWH": "other_kwh",
    }

    rows = []

    for row in df.itertuples(index=False):
        record = {
            "household_id": int(row.CUSTOMER_ID),
            "reading_timestamp": row.READING_DATETIME,
            "consumption_kwh": float(row.GENERAL_SUPPLY_KWH),
            "controlled_load_kwh": None,
            "gross_generation_kwh": None,
            "net_generation_kwh": None,
            "other_kwh": None,
            "source": "SGSC",
        }

        for csv_column, db_column in optional_columns.items():
            if csv_column in df.columns:
                value = getattr(row, csv_column)

                if pd.notna(value):
                    record[db_column] = float(value)

        rows.append(record)

    return rows


def load_energy_batch(connection, rows):
    """Insert/update one batch of hourly SGSC readings."""

    statement = text(
        """
        INSERT INTO energy_readings (
            household_id,
            reading_timestamp,
            consumption_kwh,
            controlled_load_kwh,
            gross_generation_kwh,
            net_generation_kwh,
            other_kwh,
            source
        )
        VALUES (
            :household_id,
            :reading_timestamp,
            :consumption_kwh,
            :controlled_load_kwh,
            :gross_generation_kwh,
            :net_generation_kwh,
            :other_kwh,
            :source
        )
        ON CONFLICT (household_id, reading_timestamp)
        DO UPDATE SET
            consumption_kwh = EXCLUDED.consumption_kwh,
            controlled_load_kwh = EXCLUDED.controlled_load_kwh,
            gross_generation_kwh = EXCLUDED.gross_generation_kwh,
            net_generation_kwh = EXCLUDED.net_generation_kwh,
            other_kwh = EXCLUDED.other_kwh,
            source = EXCLUDED.source;
        """
    )

    connection.execute(statement, rows)


def verify_database(engine):
    """Print SGSC database counts after ingestion."""

    with engine.connect() as connection:
        household_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM households
                WHERE source = 'SGSC';
                """
            )
        ).scalar_one()

        energy_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM energy_readings
                WHERE source = 'SGSC';
                """
            )
        ).scalar_one()

        print("\nDatabase verification")
        print("---------------------")
        print(f"SGSC households:      {household_count:,}")
        print(f"SGSC energy readings: {energy_count:,}")


def ingest_csv(input_file=DEFAULT_INPUT_FILE):
    """Load cleaned hourly SGSC data into the project database."""

    input_file = Path(input_file)

    print(f"Reading cleaned SGSC data:")
    print(input_file)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Cleaned SGSC file not found: {input_file}"
        )

    df = pd.read_csv(input_file)

    print(f"Rows loaded: {len(df):,}")

    df["READING_DATETIME"] = pd.to_datetime(
        df["READING_DATETIME"],
        errors="coerce",
    )

    for column in [
        "GENERAL_SUPPLY_KWH",
        "CONTROLLED_LOAD_KWH",
        "GROSS_GENERATION_KWH",
        "NET_GENERATION_KWH",
        "OTHER_KWH",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    validate_input(df)

    household_ids = sorted(
        df["CUSTOMER_ID"].astype(int).unique()
    )

    print(f"Households: {len(household_ids)}")

    engine = get_engine()

    # Insert households first because energy_readings has a FK.
    with engine.begin() as connection:
        load_households(
            connection,
            household_ids,
        )

    total_rows = len(df)

    for start in range(0, total_rows, BATCH_SIZE):
        end = min(
            start + BATCH_SIZE,
            total_rows,
        )

        batch = df.iloc[start:end]

        rows = prepare_energy_rows(batch)

        with engine.begin() as connection:
            load_energy_batch(
                connection,
                rows,
            )

        print(
            f"Inserted/updated "
            f"{end:,}/{total_rows:,} readings"
        )

    verify_database(engine)

    print("\nSGSC ingestion completed successfully.")


if __name__ == "__main__":
    ingest_csv()