

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

METADATA_FILE = (
    BASE_DIR / "data" / "interim" / "sgsc_selected_30_households.csv"
)

BATCH_SIZE = 5000


def get_engine():
    """Create a SQLAlchemy database engine."""
    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
    )


def validate_input(df: pd.DataFrame) -> None:
    """Validate cleaned SGSC data before database ingestion."""

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


def convert_yes_no(value):
    """Convert SGSC Y/N values to PostgreSQL boolean values."""

    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    if value == "Y":
        return True

    if value == "N":
        return False

    return None


def clean_metadata_value(value):
    """Convert pandas missing values to None for PostgreSQL."""

    if pd.isna(value):
        return None

    return str(value).strip()


def load_household_metadata():
    """Read metadata for the selected 30 SGSC households."""

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"SGSC household metadata file not found: {METADATA_FILE}"
        )

    metadata = pd.read_csv(
        METADATA_FILE,
        low_memory=False,
    )

    required_columns = {
        "CUSTOMER_ID",
        "DWELLING_TYPE_CD",
        "ASSRTD_CLIMATE_ZONE_DESC",
        "ASSRTD_ELECTRICITY_USE_GRP_CD",
        "HAS_AIRCON",
        "HAS_GENERATION",
    }

    missing = required_columns - set(metadata.columns)

    if missing:
        raise ValueError(
            "Selected household metadata is missing columns: "
            f"{sorted(missing)}"
        )

    metadata["CUSTOMER_ID"] = pd.to_numeric(
        metadata["CUSTOMER_ID"],
        errors="coerce",
    )

    if metadata["CUSTOMER_ID"].isna().any():
        raise ValueError(
            "Selected household metadata contains invalid CUSTOMER_ID values."
        )

    metadata = metadata.drop_duplicates(
        subset=["CUSTOMER_ID"],
        keep="first",
    )

    return metadata


def load_households(connection, metadata: pd.DataFrame):
    """
    Insert/update SGSC households using the existing households schema.
    """

    statement = text(
        """
        INSERT INTO households (
            household_id,
            source,
            dwelling_type,
            climate_zone,
            electricity_use_group,
            has_aircon,
            has_generation
        )
        VALUES (
            :household_id,
            :source,
            :dwelling_type,
            :climate_zone,
            :electricity_use_group,
            :has_aircon,
            :has_generation
        )
        ON CONFLICT (household_id)
        DO UPDATE SET
            source = EXCLUDED.source,
            dwelling_type = EXCLUDED.dwelling_type,
            climate_zone = EXCLUDED.climate_zone,
            electricity_use_group = EXCLUDED.electricity_use_group,
            has_aircon = EXCLUDED.has_aircon,
            has_generation = EXCLUDED.has_generation;
        """
    )

    rows = []

    for row in metadata.itertuples(index=False):
        rows.append(
            {
                "household_id": int(row.CUSTOMER_ID),
                "source": "SGSC",
                "dwelling_type": clean_metadata_value(
                    row.DWELLING_TYPE_CD
                ),
                "climate_zone": clean_metadata_value(
                    row.ASSRTD_CLIMATE_ZONE_DESC
                ),
                "electricity_use_group": clean_metadata_value(
                    row.ASSRTD_ELECTRICITY_USE_GRP_CD
                ),
                "has_aircon": convert_yes_no(
                    row.HAS_AIRCON
                ),
                "has_generation": convert_yes_no(
                    row.HAS_GENERATION
                ),
            }
        )

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
    """Verify SGSC database counts and household metadata."""

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

        metadata_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM households
                WHERE source = 'SGSC'
                  AND climate_zone IS NOT NULL
                  AND electricity_use_group IS NOT NULL;
                """
            )
        ).scalar_one()

        print("\nDatabase verification")
        print("---------------------")
        print(f"SGSC households:               {household_count:,}")
        print(f"SGSC energy readings:          {energy_count:,}")
        print(f"Households with core metadata: {metadata_count:,}")


def ingest_csv(input_file=DEFAULT_INPUT_FILE):
    """Load cleaned hourly SGSC data into the project database."""

    input_file = Path(input_file)

    print("Reading cleaned SGSC data:")
    print(input_file)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Cleaned SGSC file not found: {input_file}"
        )

    df = pd.read_csv(
        input_file,
        low_memory=False,
    )

    # Defensive cleanup in case CSV headers contain whitespace.
    df.columns = df.columns.str.strip()

    print(f"Rows loaded: {len(df):,}")

    df["CUSTOMER_ID"] = pd.to_numeric(
        df["CUSTOMER_ID"],
        errors="coerce",
    )

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
        df["CUSTOMER_ID"]
        .astype(int)
        .unique()
    )

    print(f"Households in energy dataset: {len(household_ids)}")

    metadata = load_household_metadata()

    metadata_ids = set(
        metadata["CUSTOMER_ID"]
        .astype(int)
        .tolist()
    )

    energy_ids = set(household_ids)

    missing_metadata = energy_ids - metadata_ids

    if missing_metadata:
        raise ValueError(
            "Metadata missing for these households: "
            f"{sorted(missing_metadata)}"
        )

    # Only metadata belonging to households present in the
    # cleaned energy dataset is inserted.
    metadata = metadata[
        metadata["CUSTOMER_ID"]
        .astype(int)
        .isin(energy_ids)
    ].copy()

    print(f"Households with metadata: {len(metadata)}")

    engine = get_engine()

    # households must exist first because energy_readings
    # references households through a foreign key.
    with engine.begin() as connection:
        load_households(
            connection,
            metadata,
        )

    total_rows = len(df)

    for start in range(
        0,
        total_rows,
        BATCH_SIZE,
    ):
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