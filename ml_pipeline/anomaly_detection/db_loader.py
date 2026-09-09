# anomaly_detection/db_loader.py
# ─────────────────────────────────────────────────────────────────────────────
# Loads energy data from Alka's PostgreSQL database
# Uses the dump restored from smart_energy.dump
#
# How to restore the dump on your machine:
#   createdb -U postgres smart_energy
#   pg_restore -U postgres -d smart_energy data/smart_energy.dump
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from dotenv import load_dotenv
from anomaly_detection.config import (
    HOUSEHOLD_IDS, DB_TIMESTAMP, DB_KWH, DB_HOUSEHOLD,
    COL_KWH, COL_TIMESTAMP, COL_HOUSEHOLD
)

load_dotenv(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".env"
))


def get_engine():
    """Create SQLAlchemy engine from .env credentials."""
    host = os.getenv("DB_HOST",     "localhost")
    port = os.getenv("DB_PORT",     "5432")
    name = os.getenv("DB_NAME",     "smart_energy")
    user = os.getenv("DB_USER",     "postgres")
    pw   = os.getenv("DB_PASSWORD", "postgres")
    return create_engine(
        f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{name}"
    )


def load_all_households(verbose: bool = True) -> pd.DataFrame:
    """
    Load energy readings for all 5 SGSC households.

    Applies column renaming to match pipeline expectations:
        reading_timestamp → timestamp (index)
        consumption_kwh   → kwh
        household_id      → household_id (kept)

    Returns:
        DataFrame with DatetimeIndex named 'timestamp',
        columns: kwh, controlled_load_kwh, household_id
    """
    engine  = get_engine()
    hh_list = ",".join(str(h) for h in HOUSEHOLD_IDS)

    query = f"""
        SELECT
            household_id,
            reading_timestamp,
            consumption_kwh,
            controlled_load_kwh
        FROM energy_readings
        WHERE household_id IN ({hh_list})
        ORDER BY household_id, reading_timestamp
    """

    df = pd.read_sql(
        query, engine,
        parse_dates=[DB_TIMESTAMP]
    )

    # Rename to pipeline column names
    df = df.rename(columns={
        DB_TIMESTAMP: COL_TIMESTAMP,
        DB_KWH:       COL_KWH
    })
    df = df.set_index(COL_TIMESTAMP)
    df.index.name = "timestamp"

    if verbose:
        print(f"✅ All households loaded:")
        print(f"   Total rows: {len(df):,}")
        summary = df.groupby(COL_HOUSEHOLD)[COL_KWH].agg(
            count="count", mean="mean",
            max="max",   min="min"
        ).round(4)
        print(summary.to_string())

    return df


def load_household_meta() -> pd.DataFrame:
    """Load household metadata (dwelling type, aircon, etc.)."""
    engine  = get_engine()
    hh_list = ",".join(str(h) for h in HOUSEHOLD_IDS)
    query   = f"""
        SELECT household_id, dwelling_type, climate_zone,
               electricity_use_group, has_aircon, has_generation
        FROM households
        WHERE household_id IN ({hh_list})
    """
    return pd.read_sql(query, engine)


if __name__ == "__main__":
    df = load_all_households()
    print(df.head())
    print(f"\nDate range: {df.index.min()} → {df.index.max()}")