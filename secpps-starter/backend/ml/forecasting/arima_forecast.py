"""
TRAINING/EVALUATION TRACK - 24-hour version, NO weather (the original
US-04 baseline). Real SGSC data, backtested against a real held-out day.

This file is unchanged from its originally tested and merged version --
kept exactly as-is since it's already proven and merged to main. It's
included here only so this folder listing is complete.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_series(engine, household_id: int) -> pd.Series:
    query = """
        SELECT reading_timestamp, consumption_kwh
        FROM energy_readings
        WHERE household_id = %(household_id)s
        ORDER BY reading_timestamp
    """
    df = pd.read_sql(query, engine, params={"household_id": household_id})
    if df.empty:
        raise ValueError(f"No readings found for household_id={household_id}")

    df["reading_timestamp"] = pd.to_datetime(df["reading_timestamp"])
    df = df.set_index("reading_timestamp")

    series = df["consumption_kwh"].resample("1h").mean()
    series = series.ffill().dropna()
    return series


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"mae": round(float(mae), 4), "rmse": round(float(rmse), 4), "mape": round(float(mape), 2)}


def generate_forecast(
    engine, household_id: int, horizon_hours: int = 24,
    order: tuple = (2, 1, 2), evaluate: bool = True,
) -> dict:
    series = load_series(engine, household_id)

    if evaluate:
        if len(series) < horizon_hours + 48:
            raise ValueError(f"Not enough data for household {household_id}: {len(series)} hourly points")
        train = series.iloc[:-horizon_hours]
        test = series.iloc[-horizon_hours:]
    else:
        train = series
        test = None

    model = ARIMA(train, order=order)
    fitted = model.fit()
    forecast_values = fitted.forecast(steps=horizon_hours)

    last_ts = train.index[-1]
    future_index = pd.date_range(start=last_ts, periods=horizon_hours + 1, freq="h")[1:]

    predictions = [
        {"timestamp": ts.isoformat(), "predicted_kwh": round(float(val), 4)}
        for ts, val in zip(future_index, forecast_values)
    ]

    result = {
        "household_id": household_id, "model": "ARIMA", "order": list(order),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_hours": len(train), "horizon_hours": horizon_hours,
        "predictions": predictions,
    }

    if evaluate:
        result["metrics"] = compute_metrics(test.values, forecast_values.values)
        result["actuals"] = [
            {"timestamp": ts.isoformat(), "actual_kwh": round(float(val), 4)}
            for ts, val in zip(test.index, test.values)
        ]

    return result


def save_forecast_result(result: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename = f"arima_{result['household_id']}_{result['generated_at'].replace(':', '-')}.json"
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--household-id", type=int, required=True)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--order", type=str, default="2,1,2")
    args = parser.parse_args()
    order = tuple(int(x) for x in args.order.split(","))

    engine = create_engine(DATABASE_URL)
    result = generate_forecast(engine, args.household_id, args.horizon_hours, order)

    print(f"Household {result['household_id']}: {result['train_hours']} train hours")
    print(f"Metrics: {result['metrics']}")
    print(f"First 3 predictions: {result['predictions'][:3]}")

    path = save_forecast_result(result)
    print(f"Saved to {path}")
