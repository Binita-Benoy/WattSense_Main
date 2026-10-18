
import argparse
import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)


def generate_series(start="2024-01-01", end=None, seed=42) -> pd.DataFrame:
    if end is None:
        end = pd.Timestamp.now().normalize()

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, end=end, freq="h")
    n = len(timestamps)

    hour = timestamps.hour.values
    dayofweek = timestamps.dayofweek.values
    dayofyear = timestamps.dayofyear.values

    base_load = 0.3
    daily = 0.5 * np.exp(-((hour - 8) ** 2) / 8) + 0.9 * np.exp(-((hour - 19) ** 2) / 10)
    is_weekend = (dayofweek >= 5).astype(float)
    weekly = is_weekend * 0.2
    seasonal = 0.4 * np.cos(2 * np.pi * (dayofyear - 15) / 365)
    noise = rng.normal(0, 0.08, n)
    spike_mask = rng.random(n) < 0.005
    spikes = spike_mask * rng.uniform(1.5, 3.0, n)

    consumption = base_load + daily + weekly + seasonal + noise + spikes
    consumption = np.clip(consumption, 0.02, None)

    return pd.DataFrame({"reading_timestamp": timestamps, "consumption_kwh": consumption})


def insert_household(engine, household_id: int, df: pd.DataFrame):
    with engine.begin() as conn:
        conn.execute(
            text("""INSERT INTO households (household_id, source, dwelling_type)
                     VALUES (:id, 'synthetic', 'demo_house') ON CONFLICT (household_id) DO NOTHING"""),
            {"id": household_id},
        )
        rows = [
            {"household_id": household_id, "reading_timestamp": r.reading_timestamp,
             "consumption_kwh": round(float(r.consumption_kwh), 4), "source": "synthetic"}
            for r in df.itertuples()
        ]
        conn.execute(
            text("""INSERT INTO energy_readings (household_id, reading_timestamp, consumption_kwh, source)
                     VALUES (:household_id, :reading_timestamp, :consumption_kwh, :source)
                     ON CONFLICT (household_id, reading_timestamp) DO NOTHING"""),
            rows,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--household-id", type=int, default=99999999)
    args = parser.parse_args()

    df = generate_series()
    print(f"Generated {len(df)} synthetic hourly readings from {df.reading_timestamp.min()} to {df.reading_timestamp.max()}")

    engine = create_engine(DATABASE_URL)
    insert_household(engine, args.household_id, df)
    print(f"Inserted into households/energy_readings under household_id={args.household_id}, source='synthetic'")
