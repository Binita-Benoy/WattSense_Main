"""
THE FULL SYSTEM, end to end.

Input:  household_id + latitude/longitude (consumption data already in
        energy_readings, however it got there)
Output: which model won for this household, its evaluation metrics, AND
        genuine 24-hour and 7-day forward forecasts using whichever model
        actually deserves to serve this household.

What happens automatically, in order:
  1. Backfill historical weather for this household/location if missing
     (ensure_historical_weather) -- consumption alone is enough input.
  2. Evaluate ARIMA (no weather) vs SARIMAX (with weather) on a real
     held-out window -- pick whichever actually performed better.
  3. Produce a genuine 24-hour forecast using the WINNING model:
       - if ARIMA won: pure autoregressive forward forecast, no weather needed
       - if SARIMAX won: fetch a REAL LIVE weather forecast for this
         location and use it as the exog input
  4. Same as step 3, but for 7 days instead of 24 hours.

Steps 3-4 produce genuinely future predictions -- no ground truth exists
for them yet, so (correctly) no accuracy metrics are reported alongside.
The metrics from step 2 are the ones that belong in your report.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import requests
from sqlalchemy import create_engine
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from arima_forecast import generate_forecast as evaluate_arima
from generate_weather_forecast import generate_weather_forecast as evaluate_sarimax
from ensure_weather import ensure_historical_weather

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)

EXOG_COLUMNS = ["temperature", "shortwave_radiation"]


# ---------------------------------------------------------------------
# Step 2: decide which model wins, based on real held-out evaluation
# ---------------------------------------------------------------------
def select_model(engine, household_id: int) -> dict:
    arima_eval = evaluate_arima(engine, household_id)
    sarimax_eval = evaluate_sarimax(engine, household_id)

    arima_mae = arima_eval["metrics"]["mae"]
    sarimax_mae = sarimax_eval["metrics"]["mae"]
    winner = "sarimax" if sarimax_mae < arima_mae else "arima"

    return {
        "winner": winner,
        "arima_metrics": arima_eval["metrics"],
        "sarimax_metrics": sarimax_eval["metrics"],
        "margin_pct": round(abs(arima_mae - sarimax_mae) / arima_mae * 100, 2),
    }


# ---------------------------------------------------------------------
# Live weather fetching, parameterized by user-supplied location
# ---------------------------------------------------------------------
def fetch_live_hourly_weather(latitude, longitude, hours=24) -> pd.DataFrame:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        "&hourly=temperature_2m,shortwave_radiation"
        "&forecast_days=2&timezone=Australia%2FSydney"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["time"]),
        "temperature": data["temperature_2m"],
        "shortwave_radiation": data["shortwave_radiation"],
    }).set_index("timestamp")
    now = pd.Timestamp.now().floor("h")
    return df[df.index >= now].iloc[:hours]


def fetch_live_daily_weather(latitude, longitude, days=7) -> pd.DataFrame:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        "&daily=temperature_2m_max,temperature_2m_min,shortwave_radiation_sum"
        f"&forecast_days={days}&timezone=Australia%2FSydney"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()["daily"]
    return pd.DataFrame({
        "date": pd.to_datetime(data["time"]),
        "temperature": [(mx + mn) / 2 for mx, mn in zip(data["temperature_2m_max"], data["temperature_2m_min"])],
        "shortwave_radiation": data["shortwave_radiation_sum"],
    }).set_index("date")


# ---------------------------------------------------------------------
# Step 3/4 helpers: load history at the right granularity
# ---------------------------------------------------------------------
def load_hourly_history(engine, household_id, with_weather: bool) -> pd.DataFrame:
    if with_weather:
        query = """
            SELECT e.reading_timestamp, e.consumption_kwh, w.temperature, w.shortwave_radiation
            FROM energy_readings e JOIN weather_readings w
              ON w.household_id = e.household_id AND w.reading_timestamp = e.reading_timestamp
            WHERE e.household_id = %(id)s ORDER BY e.reading_timestamp
        """
    else:
        query = """
            SELECT reading_timestamp, consumption_kwh
            FROM energy_readings WHERE household_id = %(id)s ORDER BY reading_timestamp
        """
    df = pd.read_sql(query, engine, params={"id": household_id})
    df["reading_timestamp"] = pd.to_datetime(df["reading_timestamp"])
    return df.set_index("reading_timestamp").resample("1h").mean().ffill().dropna()


def load_daily_history(engine, household_id, with_weather: bool) -> pd.DataFrame:
    if with_weather:
        query = """
            SELECT e.reading_timestamp, e.consumption_kwh, w.temperature, w.shortwave_radiation
            FROM energy_readings e JOIN weather_readings w
              ON w.household_id = e.household_id AND w.reading_timestamp = e.reading_timestamp
            WHERE e.household_id = %(id)s ORDER BY e.reading_timestamp
        """
        df = pd.read_sql(query, engine, params={"id": household_id})
        df["reading_timestamp"] = pd.to_datetime(df["reading_timestamp"])
        df = df.set_index("reading_timestamp")
        return pd.DataFrame({
            "consumption_kwh": df["consumption_kwh"].resample("1D").sum(),
            "temperature": df["temperature"].resample("1D").mean(),
            "shortwave_radiation": df["shortwave_radiation"].resample("1D").mean(),
        }).dropna()
    else:
        query = "SELECT reading_timestamp, consumption_kwh FROM energy_readings WHERE household_id = %(id)s ORDER BY reading_timestamp"
        df = pd.read_sql(query, engine, params={"id": household_id})
        df["reading_timestamp"] = pd.to_datetime(df["reading_timestamp"])
        return df.set_index("reading_timestamp")["consumption_kwh"].resample("1D").sum().to_frame().dropna()


# ---------------------------------------------------------------------
# Step 3: genuine 24-hour forward forecast, using the winning model
# ---------------------------------------------------------------------
def live_24h_forecast(engine, household_id, latitude, longitude, use_weather: bool) -> list:
    history = load_hourly_history(engine, household_id, with_weather=use_weather)

    if use_weather:
        exog_mean, exog_std = history[EXOG_COLUMNS].mean(), history[EXOG_COLUMNS].std()
        train_exog_scaled = (history[EXOG_COLUMNS] - exog_mean) / exog_std
        model = SARIMAX(history["consumption_kwh"], exog=train_exog_scaled, order=(2, 1, 2),
                         enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=500, method="powell")

        future_weather = fetch_live_hourly_weather(latitude, longitude, hours=24)
        future_exog_scaled = (future_weather[EXOG_COLUMNS] - exog_mean) / exog_std
        forecast_values = fitted.forecast(steps=len(future_weather), exog=future_exog_scaled)
        timestamps = future_weather.index
    else:
        fitted = ARIMA(history["consumption_kwh"], order=(2, 1, 2)).fit()
        forecast_values = fitted.forecast(steps=24)
        last_ts = history.index[-1]
        timestamps = pd.date_range(start=last_ts, periods=25, freq="h")[1:]

    return [{"timestamp": ts.isoformat(), "predicted_kwh": round(float(v), 4)}
            for ts, v in zip(timestamps, forecast_values)]


# ---------------------------------------------------------------------
# Step 4: genuine 7-day forward forecast, using the winning model
# ---------------------------------------------------------------------
def live_7day_forecast(engine, household_id, latitude, longitude, use_weather: bool) -> list:
    history = load_daily_history(engine, household_id, with_weather=use_weather)

    if use_weather:
        exog_mean, exog_std = history[EXOG_COLUMNS].mean(), history[EXOG_COLUMNS].std()
        train_exog_scaled = (history[EXOG_COLUMNS] - exog_mean) / exog_std
        model = SARIMAX(history["consumption_kwh"], exog=train_exog_scaled, order=(1, 1, 1),
                         seasonal_order=(1, 1, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=500, method="powell")

        future_weather = fetch_live_daily_weather(latitude, longitude, days=7)
        future_exog_scaled = (future_weather[EXOG_COLUMNS] - exog_mean) / exog_std
        forecast_values = fitted.forecast(steps=7, exog=future_exog_scaled)
        dates = future_weather.index
    else:
        model = SARIMAX(history["consumption_kwh"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                         enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=500)
        forecast_values = fitted.forecast(steps=7)
        last_date = history.index[-1]
        dates = pd.date_range(start=last_date, periods=8, freq="D")[1:]

    return [{"date": d.date().isoformat(), "predicted_kwh": round(float(v), 2)}
            for d, v in zip(dates, forecast_values)]


# ---------------------------------------------------------------------
# The full pipeline
# ---------------------------------------------------------------------
def run_full_pipeline(engine, household_id: int, latitude: float, longitude: float) -> dict:
    print(f"[1/4] Ensuring historical weather exists for household {household_id}...")
    ensure_historical_weather(engine, household_id, latitude, longitude)

    print("[2/4] Evaluating ARIMA vs SARIMAX on held-out data...")
    selection = select_model(engine, household_id)
    use_weather = selection["winner"] == "sarimax"
    print(f"       Winner: {selection['winner'].upper()} (margin: {selection['margin_pct']}%)")

    print("[3/4] Generating live 24-hour forecast...")
    forecast_24h = live_24h_forecast(engine, household_id, latitude, longitude, use_weather)

    print("[4/4] Generating live 7-day forecast...")
    forecast_7day = live_7day_forecast(engine, household_id, latitude, longitude, use_weather)

    return {
        "household_id": household_id,
        "latitude": latitude,
        "longitude": longitude,
        "model_selection": selection,
        "forecast_24h": forecast_24h,
        "forecast_7day": forecast_7day,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--household-id", type=int, required=True)
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)
    result = run_full_pipeline(engine, args.household_id, args.latitude, args.longitude)

    print(f"\n=== Household {result['household_id']} ===")
    print(f"Model selected: {result['model_selection']['winner'].upper()}")
    print(f"ARIMA metrics:   {result['model_selection']['arima_metrics']}")
    print(f"SARIMAX metrics: {result['model_selection']['sarimax_metrics']}")

    print("\n24-hour forecast (first 6 hours):")
    for p in result["forecast_24h"][:6]:
        print(f"  {p['timestamp']}: {p['predicted_kwh']} kWh")

    print("\n7-day forecast:")
    for p in result["forecast_7day"]:
        print(f"  {p['date']}: {p['predicted_kwh']} kWh")

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"full_pipeline_{args.household_id}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull result saved to {path}")
