"""
Weather-enhanced forecasting using SARIMAX with weather as an exogenous
input — this is the "weather as an extra input, not a separate model"
concept: the target is still consumption_kwh, weather columns just ride
alongside as additional predictors.

Requires Alka's energy+weather integration (weather_readings joined to
energy_readings by household_id + reading_timestamp).

IMPORTANT CAVEAT (read before trusting these numbers):
This evaluates using the REAL, ACTUAL weather for the held-out test hours
— which is fair for backtesting (proving weather correlation helps), but
NOT how a live deployment would work. A real system forecasting the
future doesn't know tomorrow's actual temperature, only a weather
FORECAST for it. Keep this distinction explicit in your report — it's
the same honesty point as the SGSC-timestamp discussion earlier.
"""
import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)

EXOG_COLUMNS = ["temperature", "humidity", "shortwave_radiation"]


def load_series_with_weather(engine, household_id: int) -> pd.DataFrame:
    """Pulls energy readings JOINed to weather readings for one household."""
    query = """
        SELECT e.reading_timestamp, e.consumption_kwh,
               w.temperature, w.humidity, w.shortwave_radiation
        FROM energy_readings e
        JOIN weather_readings w
          ON w.household_id = e.household_id
         AND w.reading_timestamp = e.reading_timestamp
        WHERE e.household_id = %(household_id)s
        ORDER BY e.reading_timestamp
    """
    df = pd.read_sql(query, engine, params={"household_id": household_id})
    if df.empty:
        raise ValueError(f"No joined energy+weather data for household_id={household_id}")

    df["reading_timestamp"] = pd.to_datetime(df["reading_timestamp"])
    df = df.set_index("reading_timestamp")
    df = df.resample("1h").mean()  # consumption: avg power over the hour = that hour's kWh
    df = df.ffill().dropna()
    return df


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"mae": round(float(mae), 4), "rmse": round(float(rmse), 4), "mape": round(float(mape), 2)}


def generate_weather_forecast(
    engine, household_id: int, horizon_hours: int = 24, order: tuple = (2, 1, 2)
) -> dict:
    df = load_series_with_weather(engine, household_id)
    if len(df) < horizon_hours + 48:
        raise ValueError(f"Not enough data for household {household_id}: {len(df)} hours")

    train, test = df.iloc[:-horizon_hours], df.iloc[-horizon_hours:]
    train_y, test_y = train["consumption_kwh"], test["consumption_kwh"]
    train_exog, test_exog = train[EXOG_COLUMNS], test[EXOG_COLUMNS]

    # Standardize exog columns (temperature ~0-40, humidity ~0-100, radiation
    # can be in the hundreds/thousands) -- wildly different scales make the
    # optimizer's job harder and are a common real cause of ConvergenceWarning.
    # Stats computed on TRAIN only, applied to both, to avoid leakage.
    exog_mean = train_exog.mean()
    exog_std = train_exog.std()
    train_exog_scaled = (train_exog - exog_mean) / exog_std
    test_exog_scaled = (test_exog - exog_mean) / exog_std

    model = SARIMAX(train_y, exog=train_exog_scaled, order=order,
                     enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=500, method="powell")

    # Using REAL test-period weather here -- see caveat in the module docstring
    forecast_values = fitted.forecast(steps=horizon_hours, exog=test_exog_scaled)

    future_index = test_y.index
    predictions = [
        {"timestamp": ts.isoformat(), "predicted_kwh": round(float(val), 4)}
        for ts, val in zip(future_index, forecast_values)
    ]

    return {
        "household_id": household_id,
        "model": "SARIMAX (weather-enhanced)",
        "exog_columns": EXOG_COLUMNS,
        "order": list(order),
        "train_hours": len(train),
        "horizon_hours": horizon_hours,
        "predictions": predictions,
        "metrics": compute_metrics(test_y.values, forecast_values.values),
        "caveat": "Evaluated using ACTUAL test-period weather, not a weather forecast -- valid for backtesting only",
    }


def generate_plain_forecast(engine, household_id: int, horizon_hours: int = 24, order: tuple = (2, 1, 2)) -> dict:
    """Same household, same split, NO weather -- for a fair side-by-side comparison."""
    df = load_series_with_weather(engine, household_id)  # reuse the join just to get identical rows
    train, test = df.iloc[:-horizon_hours], df.iloc[-horizon_hours:]
    train_y, test_y = train["consumption_kwh"], test["consumption_kwh"]

    fitted = ARIMA(train_y, order=order).fit()
    forecast_values = fitted.forecast(steps=horizon_hours)

    return {
        "model": "ARIMA (no weather)",
        "metrics": compute_metrics(test_y.values, forecast_values.values),
    }


import json


def save_forecast_result(result: dict, prefix: str = "sarimax_weather") -> str:
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    filename = f"{prefix}_{result['household_id']}.json"
    path = os.path.join(results_dir, filename)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--household-id", type=int, required=True)
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)

    plain = generate_plain_forecast(engine, args.household_id)
    weather = generate_weather_forecast(engine, args.household_id)

    print(f"Household {args.household_id}")
    print(f"  ARIMA (no weather):        {plain['metrics']}")
    print(f"  SARIMAX (with weather):    {weather['metrics']}")
    print(f"  CAVEAT: {weather['caveat']}")

    print("\n24-hour forecast (with weather), first 6 hours:")
    for p in weather["predictions"][:6]:
        print(f"  {p['timestamp']}: {p['predicted_kwh']} kWh")

    path = save_forecast_result(weather)
    print(f"\nFull 24-hour forecast saved to {path}")
