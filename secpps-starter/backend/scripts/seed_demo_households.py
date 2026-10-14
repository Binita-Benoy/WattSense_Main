"""
Demo households for EXISTING users (a year of hourly consumption, Nov 2025 to 12 Oct 2026).

Two households, so the demo can show both cases:
    99990001  "clean"      normal usage, no anomalies
    99990002  "anomalies"  same kind of usage with 4 planted events (listed in demo_anomaly_events.json)

What this script does (safe to run again: it replaces the demo rows each time):
    1. Makes backend/data/demo/demo_clean.csv and demo_anomalies.csv if they are missing (columns: timestamp,kwh).
       The data is generated with fixed random seeds, so everyone gets identical files. Delete a CSV (or pass
       --regenerate) to make it again; you can also edit the CSVs by hand.
    2. Loads them into the database (households + energy_readings, source='demo'). The original dataset
       households are NOT touched.
    3. Fetches matching historical weather from Open-Meteo for the anomaly detector (skip with --skip-weather).
    4. Optionally points two accounts at the demo households, so they become existing users on those households:
           --clean-email a@x.com --anomaly-email b@x.com
       (register the accounts first; this removes their "new user" mark and any saved upload/forecast, so the
        dashboard builds a fresh forecast from the demo data.)

Run from the backend folder, inside the container:
    docker compose exec backend python scripts/seed_demo_households.py \
        --clean-email you@example.com --anomaly-email other@example.com
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from sqlalchemy import create_engine, text

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data", "demo")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy")

START = "2025-11-01 00:00"
END = "2026-10-12 23:00"           # the latest full day when the demo was made
LATITUDE, LONGITUDE = -32.9283, 151.7817   # Newcastle NSW, same assumption as forecasting/ensure_weather.py

HOUSEHOLDS = {
    "clean":     {"id": 99990001, "csv": "demo_clean.csv",     "seed": 2025, "mean": 0.28, "events": []},
    "anomalies": {"id": 99990002, "csv": "demo_anomalies.csv", "seed": 2026, "mean": 0.31, "events": [
        # (start, hours, kWh in each of those hours, what it represents)
        ("2026-01-17 13:00", 4, (2.6, 3.4), "Summer heatwave: air-conditioning running flat out all afternoon"),
        ("2026-04-04 01:00", 4, (1.8, 2.2), "Appliance left running overnight (the house should be near standby)"),
        ("2026-07-22 18:00", 2, (4.5, 5.5), "Winter evening: oven and heaters on together, very high load"),
        ("2026-09-09 11:00", 3, (2.8, 3.3), "Unusual weekday-midday load while the house is normally quiet"),
    ]},
}


# ----------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------
def build_series(seed: int, mean: float) -> pd.Series:
    """Hourly kWh for a typical Australian home: low standby, morning and evening peaks, AC in summer, heating in winter."""
    idx = pd.date_range(START, END, freq="h")
    rng = np.random.default_rng(seed)
    h, dow, doy = idx.hour.values, idx.dayofweek.values, idx.dayofyear.values

    summer = np.clip(np.cos(2 * np.pi * (doy - 20) / 365), 0, None)    # peaks around 20 Jan
    winter = np.clip(np.cos(2 * np.pi * (doy - 200) / 365), 0, None)   # peaks around 19 Jul
    load = (0.13                                                       # standby: fridge, router, clocks
            + 0.16 * np.exp(-((h - 7.5) ** 2) / 3.0)                   # morning routine
            + 0.38 * np.exp(-((h - 19) ** 2) / 5.0)                    # evening peak
            + 0.05 * np.exp(-((h - 13) ** 2) / 10.0) * (dow >= 5)      # weekends: more at home in the day
            + 0.18 * summer * np.exp(-((h - 16) ** 2) / 12.0)          # air-conditioning, summer afternoons
            + 0.14 * winter * (np.exp(-((h - 7) ** 2) / 4.0) + np.exp(-((h - 20) ** 2) / 6.0)))  # winter heating
    load = load * rng.lognormal(0, 0.22, len(idx))
    load = load + (rng.random(len(idx)) < 0.03) * rng.uniform(0.10, 0.35, len(idx))   # kettle / oven / washer hours
    load = np.clip(load * (mean / load.mean()), 0.03, 1.2)             # normal usage never goes above 1.2 kWh
    return pd.Series(np.round(load, 3), index=idx, name="kwh")


def make_csv(name: str, regenerate: bool) -> str:
    spec = HOUSEHOLDS[name]
    path = os.path.abspath(os.path.join(DATA_DIR, spec["csv"]))
    if os.path.exists(path) and not regenerate:
        return path
    os.makedirs(DATA_DIR, exist_ok=True)
    s = build_series(spec["seed"], spec["mean"])
    rng = np.random.default_rng(spec["seed"] + 1)
    events_out = []
    for start, hours, (lo, hi), why in spec["events"]:
        when = pd.date_range(start, periods=hours, freq="h")
        s.loc[when] = np.round(rng.uniform(lo, hi, hours), 3)
        events_out.append({"start": start, "hours": hours, "kwh_per_hour": [lo, hi], "what": why})
    out = s.rename("kwh").rename_axis("timestamp").reset_index()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)
    if events_out:
        with open(os.path.join(DATA_DIR, "demo_anomaly_events.json"), "w") as f:
            json.dump({"household_id": spec["id"], "events": events_out}, f, indent=2)
    print(f"  wrote {path}: {len(out)} rows, mean {s.mean():.3f} kWh, max {s.max():.2f} kWh")
    return path


# ----------------------------------------------------------------------------------------------------------
# Database
# ----------------------------------------------------------------------------------------------------------
def load_household(engine, hid: int, csv_path: str):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    rows = [{"h": hid, "t": t.to_pydatetime(), "k": float(k)} for t, k in zip(df["timestamp"], df["kwh"])]
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM energy_readings WHERE household_id = :h"), {"h": hid})
        conn.execute(text("INSERT INTO households (household_id, source, dwelling_type) VALUES (:h, 'demo', 'demo_house') "
                          "ON CONFLICT (household_id) DO NOTHING"), {"h": hid})
        conn.execute(text("INSERT INTO energy_readings (household_id, reading_timestamp, consumption_kwh, source) "
                          "VALUES (:h, :t, :k, 'demo')"), rows)
    print(f"  household {hid}: {len(rows)} readings loaded ({df['timestamp'].min()} to {df['timestamp'].max()})")


def fetch_weather() -> pd.DataFrame:
    """Hourly weather for the whole demo period: archive API for old dates, forecast API for the last days. Null rows dropped."""
    start, end = pd.Timestamp(START).normalize(), pd.Timestamp(END).normalize()
    today = pd.Timestamp.now().normalize()
    cutoff = today - timedelta(days=8)          # the archive lags the present by a few days
    parts = []

    def get(url, a, b):
        r = requests.get(url, params={
            "latitude": LATITUDE, "longitude": LONGITUDE, "start_date": a.strftime("%Y-%m-%d"), "end_date": b.strftime("%Y-%m-%d"),
            "hourly": "temperature_2m,relative_humidity_2m,shortwave_radiation", "timezone": "Australia/Sydney"}, timeout=60)
        r.raise_for_status()
        x = r.json()["hourly"]
        return pd.DataFrame({"reading_timestamp": pd.to_datetime(x["time"]), "temperature": x["temperature_2m"],
                             "humidity": x["relative_humidity_2m"], "shortwave_radiation": x["shortwave_radiation"]})

    parts.append(get("https://archive-api.open-meteo.com/v1/archive", start, min(end, cutoff)))
    if end > cutoff:
        parts.append(get("https://api.open-meteo.com/v1/forecast", cutoff + timedelta(days=1), end))
    w = pd.concat(parts).dropna().drop_duplicates("reading_timestamp")
    return w


def load_weather(engine, ids):
    w = fetch_weather()
    with engine.begin() as conn:
        for hid in ids:
            conn.execute(text("DELETE FROM weather_readings WHERE household_id = :h"), {"h": hid})
            conn.execute(text("INSERT INTO weather_readings (household_id, reading_timestamp, temperature, humidity, shortwave_radiation, source) "
                              "VALUES (:h, :t, :tp, :hu, :sw, 'open_meteo_demo')"),
                         [{"h": hid, "t": r.reading_timestamp.to_pydatetime(), "tp": round(float(r.temperature), 2),
                           "hu": round(float(r.humidity), 2), "sw": round(float(r.shortwave_radiation), 2)} for r in w.itertuples()])
    print(f"  weather: {len(w)} hourly rows loaded for each demo household")


def assign_account(engine, email: str, hid: int):
    with engine.begin() as conn:
        user = conn.execute(text("SELECT id, household_id FROM users WHERE email = :e"), {"e": email}).fetchone()
        if user is None:
            print(f"  ! {email}: no such account. Register it first, then run this again.")
            return
        taken = conn.execute(text("SELECT email FROM users WHERE household_id = :h AND id <> :i"), {"h": hid, "i": user[0]}).fetchone()
        if taken:
            print(f"  ! household {hid} already belongs to {taken[0]}; one account per household.")
            return
        conn.execute(text("UPDATE users SET household_id = :h WHERE id = :i"), {"h": hid, "i": user[0]})
        conn.execute(text("DELETE FROM user_profiles WHERE user_id = :i"), {"i": user[0]})   # no profile row = existing user
        conn.execute(text("DELETE FROM forecast_jobs WHERE user_id = :i"), {"i": user[0]})   # drop any saved upload/forecast
    print(f"  {email}: now an existing user on household {hid} (was {user[1]})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regenerate", action="store_true", help="rebuild the CSV files even if they exist")
    p.add_argument("--skip-weather", action="store_true", help="don't fetch weather from Open-Meteo")
    p.add_argument("--clean-email", help="account that should see the clean household (99990001)")
    p.add_argument("--anomaly-email", help="account that should see the household with anomalies (99990002)")
    a = p.parse_args()

    print("Demo data files:")
    paths = {name: make_csv(name, a.regenerate) for name in HOUSEHOLDS}
    engine = create_engine(DATABASE_URL)
    print("Loading into the database:")
    for name, spec in HOUSEHOLDS.items():
        load_household(engine, spec["id"], paths[name])
    if not a.skip_weather:
        try:
            load_weather(engine, [s["id"] for s in HOUSEHOLDS.values()])
        except Exception as exc:  # noqa: BLE001
            print(f"  ! weather skipped ({type(exc).__name__}: {exc}). The anomaly check still works; run again with internet to add it.")
    if a.clean_email or a.anomaly_email:
        print("Accounts:")
        if a.clean_email:
            assign_account(engine, a.clean_email.strip(), HOUSEHOLDS["clean"]["id"])
        if a.anomaly_email:
            assign_account(engine, a.anomaly_email.strip(), HOUSEHOLDS["anomalies"]["id"])
    print("Done. Restart nothing: log in again and the dashboard shows the demo year.")


if __name__ == "__main__":
    main()