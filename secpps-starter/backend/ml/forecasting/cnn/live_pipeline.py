"""
Live pipeline for a brand-new household not in the training database.

Input: a CSV of the user's own hourly consumption (timestamp, kwh) and a
location string. Output: fine-tuned forecast at the requested horizon.

Steps:
  1. Geocode the location (Open-Meteo's free geocoding API)
  2. Fetch historical weather for the user's own data period (needed to
     build lag/rolling features -- NOT a future forecast, just history
     matching their own uploaded dates)
  3. Compute public holidays for that period (via the `holidays` library,
     since a new user has no pre-existing DB entry to look up)
  4. Build the same 24-column feature vector the global model expects,
     replicating the lag/cyclical/rolling/degree-day logic locally
  5. Load the direct multi-step global model for the requested horizon
  6. Fine-tune briefly on the user's own data
  7. Forecast the entire horizon in ONE pass (no future weather needed --
     see direct_multistep_global.py for why this matters)

HONEST LIMITATIONS, stated plainly:
  - controlled_load_kwh / gross_generation_kwh / net_generation_kwh are
    SGSC-specific metering breakdowns a new user's raw data won't have.
    These default to 0 (no solar/controlled-load signal) unless the user
    explicitly supplies them -- a real simplification, not silently
    hidden.
  - is_peak uses a standard fixed definition (7-9am, 5-9pm) as a
    reasonable default; the original dataset's exact definition was not
    independently reconfirmed here.
  - Public holidays require a country/state code (default: AU/NSW) since
    a free-text location string alone doesn't reliably map to a holiday
    calendar. Pass --country/--state to override.
  - Forecast accuracy at longer horizons (7-day, 1-month) has not been
    validated with real cross-validation the way the 24-hour model has --
    treat those numbers as provisional until tested properly.
"""
"""
Live pipeline for a brand-new household not in the training database.
Produces 24-hour, 7-day, AND 1-month forecasts in a single run.

DESIGN, and why it's split this way:
  - 24h and 7-day (168h) are both within Open-Meteo's real forecast
    window (~16 days) -- these use RECURSIVE forecasting with genuine
    live weather forecast data, reusing the already cross-validated
    recursive sequential Conv1D-LSTM model. The 7-day run's first 24
    hours ARE the 24-hour forecast -- no need to run it twice.
  - 1-month (720h) exceeds the real forecast window entirely -- no
    live weather exists that far out. This uses the separate DIRECT
    multi-step model (see direct_multistep_global.py), which never
    needs future weather at all, at the honest cost of not being able
    to react to a specific future heatwave or cold snap.

Steps:
  1. Geocode the location
  2. Fetch HISTORICAL weather for the user's own uploaded data period
     (builds lag/rolling features). Recent hours come from the forecast
     API; any older hours it no longer holds are filled from the
     Open-Meteo archive API.
  3. Fetch LIVE FORECAST weather (up to 16 days ahead) for the recursive
     24h/7-day forecast
  4. Compute public holidays (via `holidays`, since a new user has no
     DB entry to look up)
  5. Build the 24-column feature vector locally (fails loudly if any
     feature is still NaN, instead of silently producing NaN forecasts)
  6. Fine-tune the recursive model; forecast 168h forward using real
     forecast weather; the first 24h of that IS the 24h forecast
  7. Fine-tune the direct multi-step 1-month model separately; forecast
     720h in one pass (no future weather used)

HONEST LIMITATIONS, stated plainly:
  - controlled_load_kwh / gross_generation_kwh / net_generation_kwh
    default to 0 -- a new user's raw data won't have these SGSC-specific
    metering breakdowns.
  - is_peak uses a standard fixed definition (7-9am, 5-9pm).
  - Public holidays need a country/state code (default AU/NSW).
  - The 1-month forecast cannot react to any specific future weather --
    it forecasts from learned historical patterns only.
  - None of these horizons have been cross-validated for a NEW,
    out-of-sample household the way the original 30 households were --
    treat accuracy as provisional until tested properly.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from tensorflow import keras

try:
    import holidays as holidays_lib
except ImportError:
    holidays_lib = None

from features import FEATURE_COLUMNS, scale_columns, create_sequences, compute_calendar_features, set_seeds
from direct_multistep_global import LOOKBACK as DM_LOOKBACK, HORIZONS, MODEL_DIR as DM_MODEL_DIR
from simple_cnn_lstm import LOOKBACK as RECURSIVE_LOOKBACK, MODEL_PATH as RECURSIVE_MODEL_PATH

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_COLUMNS = ["temperature", "shortwave_radiation", "humidity"]
MAX_FORECAST_DAYS = 16  # Open-Meteo's real forecast limit
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 0.0001


def geocode_location(location: str) -> dict:
    resp = requests.get(GEOCODE_URL, params={"name": location, "count": 1}, timeout=15)
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        raise ValueError(f"Could not geocode location: {location!r}")
    r = results[0]
    return {"lat": r["latitude"], "lon": r["longitude"], "resolved_name": r.get("name"),
            "country_code": r.get("country_code")}


def _fetch_hourly_weather(url: str, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    resp = requests.get(url, params={
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": "temperature_2m,shortwave_radiation,relative_humidity_2m",
        "timezone": "auto",
    }, timeout=30)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    return pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "shortwave_radiation": hourly["shortwave_radiation"],
        "humidity": hourly["relative_humidity_2m"],
    })


def fetch_historical_weather(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Recent history comes from the forecast API, which only keeps a limited
    window of past data. Any hours it returns empty (typically the oldest part
    of the user's upload) are filled from the archive API, which covers all
    past dates but lags a few days behind the present -- so the two sources
    complement each other."""
    df = _fetch_hourly_weather(WEATHER_URL, lat, lon, start_date, end_date)
    missing = df[WEATHER_COLUMNS].isna().any(axis=1)
    if missing.any():
        gap_start = df.loc[missing, "timestamp"].min().strftime("%Y-%m-%d")
        gap_end = df.loc[missing, "timestamp"].max().strftime("%Y-%m-%d")
        print(f"  {int(missing.sum())} hours missing from forecast API -- filling {gap_start} to {gap_end} from archive")
        archive = _fetch_hourly_weather(ARCHIVE_URL, lat, lon, gap_start, gap_end)
        df = df.set_index("timestamp").combine_first(archive.set_index("timestamp")).reset_index()
        still_missing = int(df[WEATHER_COLUMNS].isna().any(axis=1).sum())
        if still_missing:
            print(f"  WARNING: {still_missing} hours still missing after archive fill")
    return df


def fetch_forecast_weather(lat: float, lon: float, forecast_days: int = MAX_FORECAST_DAYS) -> pd.DataFrame:
    """REAL future weather forecast, not history -- capped at Open-Meteo's actual limit."""
    forecast_days = min(forecast_days, MAX_FORECAST_DAYS)
    resp = requests.get(WEATHER_URL, params={
        "latitude": lat, "longitude": lon, "forecast_days": forecast_days,
        "hourly": "temperature_2m,shortwave_radiation,relative_humidity_2m",
        "timezone": "auto",
    }, timeout=30)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    return pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "shortwave_radiation": hourly["shortwave_radiation"],
        "humidity": hourly["relative_humidity_2m"],
    }).set_index("timestamp")


def get_holiday_flags(dates, country_code: str = "AU", state: str = "NSW") -> np.ndarray:
    if holidays_lib is None:
        return np.zeros(len(dates))
    years = sorted(set(pd.DatetimeIndex(dates).year))
    try:
        cal = holidays_lib.country_holidays(country_code, subdiv=state, years=years)
    except Exception:
        cal = holidays_lib.country_holidays(country_code, years=years)
    return np.array([1 if pd.Timestamp(d).date() in cal else 0 for d in dates])


def daily_totals(timestamps, hourly_kwh, n_days: int) -> list:
    """Sum hourly predictions into consecutive 24-hour blocks, one per day.
    Blocks start at the first forecast hour, so when the uploaded data ends at
    23:00 (as usual) each block is exactly one calendar day."""
    days = []
    for d in range(n_days):
        block = hourly_kwh[d * 24:(d + 1) * 24]
        start = timestamps[d * 24]
        days.append({"day": d + 1, "date": start.strftime("%Y-%m-%d"),
                     "predicted_kwh": round(float(np.sum(block)), 3)})
    return days


def build_live_feature_vector(user_df: pd.DataFrame, weather_df: pd.DataFrame,
                               country_code: str = "AU", state: str = "NSW") -> pd.DataFrame:
    df = user_df.merge(weather_df, on="timestamp", how="left").set_index("timestamp").sort_index()
    # ffill covers small gaps; bfill covers a few empty hours at the very start
    df[WEATHER_COLUMNS] = df[WEATHER_COLUMNS].ffill().bfill()

    df["lag_1h"] = df["kwh"].shift(1)
    df["lag_24h"] = df["kwh"].shift(24)
    df["lag_48h"] = df["kwh"].shift(48)
    df["lag_168h"] = df["kwh"].shift(168)

    calendar = df.index.map(compute_calendar_features)
    df["hour_sin"] = [c["hour_sin"] for c in calendar]
    df["hour_cos"] = [c["hour_cos"] for c in calendar]
    df["dow_sin"] = [c["dow_sin"] for c in calendar]
    df["dow_cos"] = [c["dow_cos"] for c in calendar]
    df["season"] = [c["season"] for c in calendar]

    df["rolling_7d_mean"] = df["kwh"].rolling(168, min_periods=1).mean()
    df["rolling_7d_std"] = df["kwh"].rolling(168, min_periods=1).std().fillna(0)
    df["rolling_30d_mean"] = df["kwh"].rolling(720, min_periods=1).mean()
    df["rolling_30d_std"] = df["kwh"].rolling(720, min_periods=1).std().fillna(0)

    hour_of_day = df.index.hour
    df["is_peak"] = np.where(((hour_of_day >= 7) & (hour_of_day < 9)) |
                              ((hour_of_day >= 17) & (hour_of_day < 21)), 1, 0)

    base_temp = 18.0
    df["heating_degree"] = (base_temp - df["temperature"]).clip(lower=0)
    df["cooling_degree"] = (df["temperature"] - base_temp).clip(lower=0)
    df["is_public_holiday"] = get_holiday_flags(df.index, country_code, state)

    df["controlled_load_kwh"] = 0.0
    df["gross_generation_kwh"] = 0.0
    df["net_generation_kwh"] = 0.0

    out = df.dropna(subset=["lag_168h"])[FEATURE_COLUMNS]
    bad = out.columns[out.isna().any()].tolist()
    if bad:
        raise ValueError(f"Feature columns still contain NaN (the models would output NaN): {bad}")
    return out


def build_live_next_row(next_kwh_scaled, ts, history_df, forecast_weather_df, scalers, last_rolling_and_peak,
                         country_code, state):
    """Same role as features.py's build_next_row, but pulls weather from
    a REAL live forecast for future timestamps, not historical actuals."""
    calendar = compute_calendar_features(ts)

    def real_lag_scaled(hours_back, col):
        target_time = ts - pd.Timedelta(hours=hours_back)
        raw_val = history_df.loc[target_time, "kwh"] if target_time in history_df.index else history_df["kwh"].iloc[-1]
        return scalers[col].transform([[raw_val]])[0, 0]

    if ts in forecast_weather_df.index:
        temp = forecast_weather_df.loc[ts, "temperature"]
        rad = forecast_weather_df.loc[ts, "shortwave_radiation"]
        hum = forecast_weather_df.loc[ts, "humidity"]
    else:
        temp, rad, hum = forecast_weather_df["temperature"].iloc[-1], forecast_weather_df["shortwave_radiation"].iloc[-1], forecast_weather_df["humidity"].iloc[-1]

    heating_degree = max(0, 18.0 - temp)
    cooling_degree = max(0, temp - 18.0)
    holiday_flag = get_holiday_flags([ts], country_code, state)[0]

    return np.array([
        next_kwh_scaled, next_kwh_scaled,
        real_lag_scaled(24, "lag_24h"), real_lag_scaled(48, "lag_48h"), real_lag_scaled(168, "lag_168h"),
        scalers["hour_sin"].transform([[calendar["hour_sin"]]])[0, 0],
        scalers["hour_cos"].transform([[calendar["hour_cos"]]])[0, 0],
        scalers["dow_sin"].transform([[calendar["dow_sin"]]])[0, 0],
        scalers["dow_cos"].transform([[calendar["dow_cos"]]])[0, 0],
        last_rolling_and_peak["rolling_7d_mean"], last_rolling_and_peak["rolling_7d_std"],
        last_rolling_and_peak["rolling_30d_mean"], last_rolling_and_peak["rolling_30d_std"],
        last_rolling_and_peak["is_peak"],
        scalers["season"].transform([[calendar["season"]]])[0, 0],
        scalers["temperature"].transform([[temp]])[0, 0],
        scalers["shortwave_radiation"].transform([[rad]])[0, 0],
        scalers["humidity"].transform([[hum]])[0, 0],
        scalers["controlled_load_kwh"].transform([[0.0]])[0, 0],
        scalers["gross_generation_kwh"].transform([[0.0]])[0, 0],
        scalers["net_generation_kwh"].transform([[0.0]])[0, 0],
        scalers["heating_degree"].transform([[heating_degree]])[0, 0],
        scalers["cooling_degree"].transform([[cooling_degree]])[0, 0],
        scalers["is_public_holiday"].transform([[holiday_flag]])[0, 0],
    ])


def run_live_forecast_all_horizons(user_csv_path: str, location: str,
                                    country_code: str = "AU", state: str = "NSW") -> dict:
    user_df = pd.read_csv(user_csv_path)
    user_df["timestamp"] = pd.to_datetime(user_df["timestamp"])
    if "kwh" not in user_df.columns:
        raise ValueError("Input CSV must have columns: timestamp, kwh")

    print(f"Geocoding {location!r}...")
    geo = geocode_location(location)
    print(f"  -> resolved to {geo['resolved_name']} ({geo['lat']}, {geo['lon']})")

    start_date = user_df["timestamp"].min().strftime("%Y-%m-%d")
    end_date = user_df["timestamp"].max().strftime("%Y-%m-%d")
    print(f"Fetching historical weather for {start_date} to {end_date}...")
    hist_weather_df = fetch_historical_weather(geo["lat"], geo["lon"], start_date, end_date)

    print(f"Fetching live weather forecast (up to {MAX_FORECAST_DAYS} days)...")
    forecast_weather_df = fetch_forecast_weather(geo["lat"], geo["lon"])

    print("Building feature vector...")
    feature_df = build_live_feature_vector(user_df, hist_weather_df, country_code, state)

    result = {"location": geo, "generated_at": datetime.now(timezone.utc).isoformat(), "horizons": {}}

    # --- 24h + 7-day: recursive, using REAL live forecast weather ---
    print("\n=== Recursive forecast (24h + 7-day), using live weather forecast ===")
    set_seeds()
    recursive_model = keras.models.load_model(RECURSIVE_MODEL_PATH)
    recursive_model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss="mse")

    scaled, scalers = scale_columns(feature_df)
    X, y = create_sequences(scaled, RECURSIVE_LOOKBACK)
    ft_split = max(1, int(len(X) * 0.9)) if len(X) >= 10 else len(X)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    recursive_model.fit(X[:ft_split], y[:ft_split],
                         validation_data=(X[ft_split:], y[ft_split:]) if ft_split < len(X) else None,
                         epochs=FINE_TUNE_EPOCHS, batch_size=8,
                         callbacks=[early_stop] if ft_split < len(X) else [], verbose=0)

    kwh_scaler = scalers["kwh"]
    window = scaled[-RECURSIVE_LOOKBACK:, :].copy()
    last_rolling_and_peak = {
        col: scaled[-1, FEATURE_COLUMNS.index(col)]
        for col in ["rolling_7d_mean", "rolling_7d_std", "rolling_30d_mean", "rolling_30d_std", "is_peak"]
    }
    last_timestamp = feature_df.index[-1]
    recursive_horizon = 168  # 7 days; the first 24h of this doubles as the 24h forecast

    scaled_predictions = []
    for step in range(recursive_horizon):
        ts = last_timestamp + pd.Timedelta(hours=step + 1)
        x_input = window.reshape((1, RECURSIVE_LOOKBACK, len(FEATURE_COLUMNS)))
        next_kwh_scaled = recursive_model.predict(x_input, verbose=0)[0, 0]
        scaled_predictions.append(next_kwh_scaled)
        next_row = build_live_next_row(next_kwh_scaled, ts, feature_df, forecast_weather_df,
                                        scalers, last_rolling_and_peak, country_code, state)
        window = np.vstack([window[1:], next_row])

    recursive_kwh = kwh_scaler.inverse_transform(np.array(scaled_predictions).reshape(-1, 1)).flatten()
    recursive_kwh = np.clip(recursive_kwh, 0, None)  # consumption can't be negative
    recursive_timestamps = pd.date_range(last_timestamp + pd.Timedelta(hours=1), periods=recursive_horizon, freq="h")

    hourly_24 = [{"timestamp": ts.isoformat(), "predicted_kwh": round(float(v), 4)}
                 for ts, v in zip(recursive_timestamps[:24], recursive_kwh[:24])]
    daily_7 = daily_totals(recursive_timestamps, recursive_kwh, 7)
    result["horizons"]["24h"] = {
        "unit": "kWh per hour", "predictions": hourly_24,
        "total_kwh": round(float(recursive_kwh[:24].sum()), 3),
        "method": "recursive with live weather forecast",
    }
    result["horizons"]["7day"] = {
        "unit": "kWh per day", "predictions": daily_7,
        "total_kwh": round(float(recursive_kwh.sum()), 3),
        "method": "recursive with live weather forecast, summed per day",
    }

    # --- 1-month: direct multi-step, no future weather needed ---
    print("\n=== Direct multi-step forecast (1-month) ===")
    month_model_path = os.path.join(DM_MODEL_DIR, "global_direct_multistep_1month.keras")
    if os.path.exists(month_model_path):
        set_seeds()
        month_model = keras.models.load_model(month_model_path)
        month_model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss="mse")

        horizon = HORIZONS["1month"]
        X_dm, y_dm = [], []
        for i in range(len(scaled) - DM_LOOKBACK - horizon):
            X_dm.append(scaled[i:i + DM_LOOKBACK, :])
            y_dm.append(scaled[i + DM_LOOKBACK: i + DM_LOOKBACK + horizon, 0])

        if len(X_dm) > 0:
            X_dm, y_dm = np.array(X_dm), np.array(y_dm)
            dm_split = max(1, int(len(X_dm) * 0.9)) if len(X_dm) >= 10 else len(X_dm)
            month_model.fit(X_dm[:dm_split], y_dm[:dm_split],
                             validation_data=(X_dm[dm_split:], y_dm[dm_split:]) if dm_split < len(X_dm) else None,
                             epochs=FINE_TUNE_EPOCHS, batch_size=8,
                             callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)] if dm_split < len(X_dm) else [],
                             verbose=0)
            last_window = scaled[-DM_LOOKBACK:, :].reshape(1, DM_LOOKBACK, len(FEATURE_COLUMNS))
            scaled_month = month_model.predict(last_window, verbose=0).flatten()
            month_kwh = kwh_scaler.inverse_transform(scaled_month.reshape(-1, 1)).flatten()
            month_kwh = np.clip(month_kwh, 0, None)
            month_timestamps = pd.date_range(last_timestamp + pd.Timedelta(hours=1), periods=horizon, freq="h")
            n_days = horizon // 24  # 720h -> 30 days
            result["horizons"]["1month"] = {
                "unit": "kWh per day", "predictions": daily_totals(month_timestamps, month_kwh, n_days),
                "total_kwh": round(float(month_kwh.sum()), 3),
                "method": "direct multi-step, no future weather, summed per day",
            }
        else:
            result["horizons"]["1month"] = {"error": "Not enough historical data to build 1-month training sequences."}
    else:
        result["horizons"]["1month"] = {
            "error": f"No 1-month model found at {month_model_path}. "
                     f"Run: direct_multistep_global.py --horizon 1month first."
        }

    result["limitations"] = [
        "controlled_load_kwh/gross_generation_kwh/net_generation_kwh defaulted to 0 -- no solar/controlled-load "
        "signal available for this new household.",
        "24h and 7-day forecasts use real live weather forecast data (Open-Meteo, up to 16 days out).",
        "1-month forecast uses no future weather at all -- it cannot react to a specific future heatwave or "
        "cold snap, only learned historical patterns.",
        f"Public holidays computed for {country_code}/{state} -- override if incorrect for this location.",
        "None of these horizons have been cross-validated for a new, out-of-sample household -- treat accuracy "
        "as provisional until tested properly.",
    ]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--user-csv", required=True, help="CSV with columns: timestamp, kwh")
    parser.add_argument("--location", required=True, help="e.g. 'Newcastle, Australia'")
    parser.add_argument("--country", default="AU")
    parser.add_argument("--state", default="NSW")
    args = parser.parse_args()

    result = run_live_forecast_all_horizons(args.user_csv, args.location, args.country, args.state)

    print(f"\nLocation: {result['location']['resolved_name']}")
    for horizon_name, data in result["horizons"].items():
        if "error" in data:
            print(f"\n{horizon_name}: ERROR - {data['error']}")
            continue
        print(f"\n{horizon_name}: {len(data['predictions'])} predictions ({data['unit']}), "
              f"total {data['total_kwh']} kWh -- {data['method']}")
        for p in data["predictions"]:
            label = p["timestamp"] if "timestamp" in p else f"Day {p['day']:>2} {p['date']}"
            print(f"  {label}: {p['predicted_kwh']}")

    print(f"\nLimitations:")
    for lim in result["limitations"]:
        print(f"  - {lim}")

    out_path = "live_forecast_all_horizons.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved full result to {out_path}")