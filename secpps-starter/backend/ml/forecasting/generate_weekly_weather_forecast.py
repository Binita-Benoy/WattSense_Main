"""
TRAINING/EVALUATION TRACK - 7-day version.
Real SGSC household data, real historical weather, backtested against a
real held-out week -> produces genuine MAE/RMSE/MAPE for your report.
This is the weekly counterpart to generate_weather_forecast.py (which
does the same thing at 24-hour/hourly granularity).

NOT for the live demo -- see generate_live_24h_forecast.py and
generate_live_weekly_forecast.py for that track.
"""
import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)

EXOG_COLUMNS = ["temperature", "humidity", "shortwave_radiation"]


def load_daily_series_with_weather(engine, household_id: int) -> pd.DataFrame:
    query = """
        SELECT e.reading_timestamp, e.consumption_kwh,
               w.temperature, w.humidity, w.shortwave_radiation
        FROM energy_readings e
        JOIN weather_readings w
          ON w.household_id = e.household_id AND w.reading_timestamp = e.reading_timestamp
        WHERE e.household_id = %(household_id)s
        ORDER BY e.reading_timestamp
    """
    df = pd.read_sql(query, engine, params={"household_id": household_id})
    if df.empty:
        raise ValueError(f"No joined energy+weather data for household_id={household_id}")

    df["reading_timestamp"] = pd.to_datetime(df["reading_timestamp"])
    df = df.set_index("reading_timestamp")

    daily = pd.DataFrame({
        "consumption_kwh": df["consumption_kwh"].resample("1D").sum(),   # energy: SUM over the day
        "temperature": df["temperature"].resample("1D").mean(),          # weather: MEAN over the day
        "humidity": df["humidity"].resample("1D").mean(),
        "shortwave_radiation": df["shortwave_radiation"].resample("1D").mean(),
    }).dropna()

    return daily


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"mae": round(float(mae), 4), "rmse": round(float(rmse), 4), "mape": round(float(mape), 2)}


def generate_weekly_weather_forecast(
    engine, household_id: int, horizon_days: int = 7,
    order: tuple = (1, 1, 1), seasonal_order: tuple = (1, 1, 1, 7),
) -> dict:
    df = load_daily_series_with_weather(engine, household_id)
    min_days = horizon_days + 60
    if len(df) < min_days:
        raise ValueError(f"Not enough daily data for household {household_id}: {len(df)} days (need {min_days})")

    train, test = df.iloc[:-horizon_days], df.iloc[-horizon_days:]
    train_y, test_y = train["consumption_kwh"], test["consumption_kwh"]
    train_exog, test_exog = train[EXOG_COLUMNS], test[EXOG_COLUMNS]

    # Standardize exog using TRAIN stats only (avoid leakage) -- same
    # reasoning as the hourly version.
    exog_mean, exog_std = train_exog.mean(), train_exog.std()
    train_exog_scaled = (train_exog - exog_mean) / exog_std
    test_exog_scaled = (test_exog - exog_mean) / exog_std

    model = SARIMAX(train_y, exog=train_exog_scaled, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=500, method="powell")

    forecast_values = fitted.forecast(steps=horizon_days, exog=test_exog_scaled)

    predictions = [
        {"date": d.date().isoformat(), "predicted_kwh": round(float(v), 2)}
        for d, v in zip(test_y.index, forecast_values)
    ]

    return {
        "household_id": household_id,
        "model": "SARIMAX weekly (weather-enhanced)",
        "exog_columns": EXOG_COLUMNS,
        "train_days": len(train),
        "horizon_days": horizon_days,
        "predictions": predictions,
        "metrics": compute_metrics(test_y.values, forecast_values.values),
        "caveat": "Evaluated using ACTUAL test-period weather -- valid for backtesting only",
    }


def generate_weekly_plain_forecast(engine, household_id: int, horizon_days: int = 7,
                                    order: tuple = (1, 1, 1), seasonal_order: tuple = (1, 1, 1, 7)) -> dict:
    """No weather -- for a fair side-by-side comparison against the weather-enhanced version."""
    df = load_daily_series_with_weather(engine, household_id)
    train, test = df.iloc[:-horizon_days], df.iloc[-horizon_days:]
    train_y, test_y = train["consumption_kwh"], test["consumption_kwh"]

    fitted = SARIMAX(train_y, order=order, seasonal_order=seasonal_order,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=500)
    forecast_values = fitted.forecast(steps=horizon_days)

    return {"model": "SARIMA weekly (no weather)", "metrics": compute_metrics(test_y.values, forecast_values.values)}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--household-id", type=int, required=True)
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)
    plain = generate_weekly_plain_forecast(engine, args.household_id)
    weather = generate_weekly_weather_forecast(engine, args.household_id)

    print(f"Household {args.household_id}")
    print(f"  SARIMA weekly (no weather):     {plain['metrics']}")
    print(f"  SARIMAX weekly (with weather):  {weather['metrics']}")
    print(f"  CAVEAT: {weather['caveat']}")
    print("\n7-day forecast (with weather):")
    for p in weather["predictions"]:
        print(f"  {p['date']}: {p['predicted_kwh']} kWh")

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"sarimax_weekly_weather_{args.household_id}.json")
    with open(path, "w") as f:
        json.dump(weather, f, indent=2)
    print(f"\nFull forecast saved to {path}")
