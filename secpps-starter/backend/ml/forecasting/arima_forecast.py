"""
Reusable ARIMA forecasting logic for US-04.

This replaces the CLI-only version of arima_baseline.py with a function
that anyone else's code can import and call directly:
  - Jazim's API layer (US-07) calls generate_forecast() to serve live predictions
  - Binita's comparison script (US-08) calls it across multiple households
  - This file's own __main__ block still works as a CLI for manual testing

Acceptance criteria satisfied here:
  - ARIMA receives a household's historical data       -> load_series()
  - ARIMA produces future predictions                   -> generate_forecast()
  - Predictions have timestamps                         -> pd.date_range() in generate_forecast()
  - MAE, RMSE, MAPE are calculated                       -> compute_metrics()
  - Results are reproducible                             -> no randomness in ARIMA fitting; see test file
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
    """Pulls raw readings for one household and resamples to hourly.

    NOTE: minimal inline cleaning (resample + forward-fill) until Alka's
    clean_readings table/view exists — swap the query source then, nothing
    else in this file needs to change.
    """
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
    engine,
    household_id: int,
    horizon_hours: int = 24,
    order: tuple = (2, 1, 2),
    evaluate: bool = True,
) -> dict:
    """
    Core reusable function. Returns a plain dict (JSON-serialisable) so it's
    equally easy to call from a script, a test, or a FastAPI endpoint.

    If evaluate=True, holds out the last `horizon_hours` of REAL data to
    score against (what your ARIMA baseline script already did).
    If evaluate=False, forecasts truly into the future beyond the data
    you have (no metrics possible, since there's nothing to compare against).
    """
    series = load_series(engine, household_id)

    if evaluate:
        if len(series) < horizon_hours + 48:
            raise ValueError(
                f"Not enough data for household {household_id}: {len(series)} hourly points"
            )
        train = series.iloc[:-horizon_hours]
        test = series.iloc[-horizon_hours:]
    else:
        train = series
        test = None

    model = ARIMA(train, order=order)
    fitted = model.fit()
    forecast_values = fitted.forecast(steps=horizon_hours)

    # Attach real timestamps: the next `horizon_hours` hours after the
    # last point the model was trained on.
    last_ts = train.index[-1]
    future_index = pd.date_range(start=last_ts, periods=horizon_hours + 1, freq="h")[1:]

    predictions = [
        {"timestamp": ts.isoformat(), "predicted_kwh": round(float(val), 4)}
        for ts, val in zip(future_index, forecast_values)
    ]

    result = {
        "household_id": household_id,
        "model": "ARIMA",
        "order": list(order),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_hours": len(train),
        "horizon_hours": horizon_hours,
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
    """
    Saves a forecast result to a JSON file under results/.

    NOTE: this is a simple file-based save to unblock US-04 today. Longer
    term, this probably becomes a `forecast_results` DB table so the API
    can query history — but that's a schema change, which is Alka's domain.
    Raise it with her before adding a new table, rather than doing it
    unilaterally on the shared database.
    """
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
