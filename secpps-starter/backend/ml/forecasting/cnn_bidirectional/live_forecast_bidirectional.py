"""
Live pipeline for a brand-new household using the BIDIRECTIONAL CNN-LSTM variant.
One run produces 24-hour, 7-day AND 1-month forecasts into a single JSON file.

Input : a CSV with columns `timestamp, kwh` (hourly preferred; sub-hourly is
        summed to hourly) and a free-text location, e.g. "Newcastle, Australia".
Output: live_forecast_bidirectional.json

How the horizons are produced (same split as the validated pipeline):
  - 24h + 7-day : RECURSIVE forecast, 168 steps, using Open-Meteo weather for the
                  forecast window. The first 24 steps ARE the 24h forecast.
                  Model: bidirectional_cnn_lstm_pooled.keras (fine-tuned on the
                  household's own data first).
  - 1-month     : DIRECT multi-step (one pass -> 720 hours), NO weather.
                  Model: bidirectional_direct_multistep_1month.keras (fine-tuned first).

Steps: geocode -> historical weather for the CSV period -> weather for the
forecast window -> public holidays -> 24-column feature vector -> fine-tune ->
forecast -> JSON.

Usage:
    docker compose exec -T backend python ml/forecasting/cnn_bidirectional/live_forecast_bidirectional.py \
        --user-csv /app/data/household.csv --location "Newcastle, Australia" --country AU --state NSW

Differences from the earlier sequential live pipeline (all flagged in the JSON
"notes" so nothing is hidden):
  1. Recursive lags use the model's OWN earlier predictions once they are
     available (lag_1h/24h/48h/168h). The old builder fell back to the last
     real value, so every 7-day lag beyond the first day was frozen.
  2. is_peak is computed from each future timestamp (it is a pure function of
     the hour); rolling 7d/30d stats are held at their last real values, as in
     the validated CV runs.
  3. Weather for the forecast window comes from the right Open-Meteo source for
     those dates (archive if the dates are in the past, forecast API if recent
     or future). Hours with no weather are imputed from the household's recent
     same-hour average and COUNTED in the output, never silently filled.
  4. Fine-tuning uses Huber loss, like the training scripts.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from tensorflow import keras

# Sibling-folder import shim (features.py / simple_cnn_lstm.py live in sibling folders).
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sib in sorted(os.listdir(_PARENT)):
    _p = os.path.join(_PARENT, _sib)
    if os.path.isdir(_p) and _p not in sys.path and any(
        os.path.exists(os.path.join(_p, f)) for f in ("features.py", "simple_cnn_lstm.py", "scaler_io.py")
    ):
        sys.path.append(_p)

try:
    import holidays as holidays_lib
except ImportError:
    holidays_lib = None

from features import FEATURE_COLUMNS, scale_columns, create_sequences, compute_calendar_features, set_seeds
from bidirectional_cnn_lstm_multihorizon import LOOKBACK as REC_LOOKBACK, MODEL_PATH as REC_MODEL_PATH
from bidirectional_direct_multistep_1month import (
    LOOKBACK as DM_LOOKBACK,
    HORIZON as MONTH_HORIZON,
    MODEL_PATH as MONTH_MODEL_PATH,
    create_direct_multistep_sequences,
)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_COLUMNS = ["temperature", "shortwave_radiation", "humidity"]
FORECAST_API_FUTURE_DAYS = 15     # Open-Meteo forecast limit is ~16 days
FORECAST_API_PAST_DAYS = 90       # forecast endpoint only keeps ~92 days of past
ARCHIVE_LAG_DAYS = 6              # archive lags the present by a few days
RECURSIVE_HORIZON = 168
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 0.0001

# Cross-validation evidence for this model (household 8162493, pool 6, 5-fold rolling origin).
# Edit the 1-month entry once its CV finishes.
VALIDATION_EVIDENCE = {
    "24h": "5-fold CV on one household: MAPE 16.36% (+/-6.24), MAE 0.0254, RMSE 0.0369",
    "7day": "5-fold CV on one household: MAPE 23.93% (+/-6.49), MAE 0.0457, RMSE 0.0648",
    "1month": "CV pending; single split on one household: MAPE 18.27%, MAE 0.0388, RMSE 0.058",
}


# ----------------------------------------------------------------------------
# Input + location + weather
# ----------------------------------------------------------------------------
def load_user_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"timestamp", "kwh"}.issubset(df.columns):
        raise ValueError("Input CSV must have columns: timestamp, kwh")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    s = df.set_index("timestamp")["kwh"].astype(float).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    hourly = s.resample("h").sum(min_count=1)       # sums sub-hourly readings, keeps gaps as NaN
    gaps = int(hourly.isna().sum())
    if gaps:
        print(f"  {gaps} missing hour(s) in the CSV -- interpolating gaps of up to 6h")
        hourly = hourly.interpolate(limit=6)
        left = int(hourly.isna().sum())
        if left:
            print(f"  WARNING: {left} hour(s) still missing; dropping the rows before the last gap")
            last_gap = hourly[hourly.isna()].index.max()
            hourly = hourly.loc[last_gap + pd.Timedelta(hours=1):]
    out = hourly.rename("kwh").reset_index().rename(columns={"index": "timestamp"})
    out.columns = ["timestamp", "kwh"]
    return out


def geocode_location(location: str) -> dict:
    resp = requests.get(GEOCODE_URL, params={"name": location, "count": 1}, timeout=15)
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        raise ValueError(f"Could not geocode location: {location!r}")
    r = results[0]
    return {"lat": r["latitude"], "lon": r["longitude"], "resolved_name": r.get("name"),
            "country_code": r.get("country_code")}


def _fetch_hourly(url: str, lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    resp = requests.get(url, params={
        "latitude": lat, "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"), "end_date": end.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,shortwave_radiation,relative_humidity_2m",
        "timezone": "auto",
    }, timeout=60)
    resp.raise_for_status()
    h = resp.json()["hourly"]
    return pd.DataFrame({
        "temperature": h["temperature_2m"],
        "shortwave_radiation": h["shortwave_radiation"],
        "humidity": h["relative_humidity_2m"],
    }, index=pd.to_datetime(h["time"]))


def fetch_weather(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Hourly weather for [start, end], picking the right Open-Meteo source per date:
    archive for old dates, forecast API for recent/future ones (capped at its real
    limits). Returns a frame indexed hourly over the full range; hours that no
    source could provide are NaN (callers decide how to handle them)."""
    start_d, end_d = start.normalize(), end.normalize()
    today = pd.Timestamp.now().normalize()
    archive_cutoff = today - pd.Timedelta(days=ARCHIVE_LAG_DAYS)
    frames = []
    if start_d <= archive_cutoff:
        frames.append(_fetch_hourly(ARCHIVE_URL, lat, lon, start_d, min(end_d, archive_cutoff)))
    f_start = max(start_d, archive_cutoff + pd.Timedelta(days=1), today - pd.Timedelta(days=FORECAST_API_PAST_DAYS))
    f_end = min(end_d, today + pd.Timedelta(days=FORECAST_API_FUTURE_DAYS))
    if f_start <= f_end:
        frames.append(_fetch_hourly(FORECAST_URL, lat, lon, f_start, f_end))
    full_index = pd.date_range(start.floor("h"), end.floor("h"), freq="h")
    if not frames:
        return pd.DataFrame(index=full_index, columns=WEATHER_COLUMNS, dtype=float)
    combined = pd.concat(frames).groupby(level=0).first()   # first() skips NaN -> sources complement each other
    return combined.reindex(full_index)


# ----------------------------------------------------------------------------
# Holidays + features
# ----------------------------------------------------------------------------
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
    days = []
    for d in range(n_days):
        block = hourly_kwh[d * 24:(d + 1) * 24]
        days.append({"day": d + 1, "date": timestamps[d * 24].strftime("%Y-%m-%d"),
                     "predicted_kwh": round(float(np.sum(block)), 3)})
    return days


def build_live_feature_vector(user_df: pd.DataFrame, weather_df: pd.DataFrame,
                               country_code: str, state: str) -> pd.DataFrame:
    df = user_df.set_index("timestamp").sort_index().join(weather_df, how="left")
    df[WEATHER_COLUMNS] = df[WEATHER_COLUMNS].ffill().bfill()
    if df[WEATHER_COLUMNS].isna().any().any():
        raise ValueError("No historical weather could be fetched for the CSV period -- check location/dates/network.")

    for lag in (1, 24, 48, 168):
        df[f"lag_{lag}h"] = df["kwh"].shift(lag)

    cal = df.index.map(compute_calendar_features)
    for key in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "season"):
        df[key] = [c[key] for c in cal]

    df["rolling_7d_mean"] = df["kwh"].rolling(168, min_periods=1).mean()
    df["rolling_7d_std"] = df["kwh"].rolling(168, min_periods=1).std().fillna(0)
    df["rolling_30d_mean"] = df["kwh"].rolling(720, min_periods=1).mean()
    df["rolling_30d_std"] = df["kwh"].rolling(720, min_periods=1).std().fillna(0)
    df["is_peak"] = [is_peak_hour(h) for h in df.index.hour]

    df["heating_degree"] = (18.0 - df["temperature"]).clip(lower=0)
    df["cooling_degree"] = (df["temperature"] - 18.0).clip(lower=0)
    df["is_public_holiday"] = get_holiday_flags(df.index, country_code, state)
    for col in ("controlled_load_kwh", "gross_generation_kwh", "net_generation_kwh"):
        df[col] = 0.0

    out = df.dropna(subset=["lag_168h"])[FEATURE_COLUMNS]
    bad = out.columns[out.isna().any()].tolist()
    if bad:
        raise ValueError(f"Feature columns still contain NaN (the models would output NaN): {bad}")
    return out


def is_peak_hour(hour: int) -> int:
    return int((7 <= hour < 9) or (17 <= hour < 21))


# ----------------------------------------------------------------------------
# Recursive step
# ----------------------------------------------------------------------------
def build_live_next_row(next_kwh_scaled, ts, kwh_map, last_kwh, weather_row, scalers, static,
                         country_code, state, counters):
    """One new scaled feature row for timestamp ts. kwh_map holds real history
    PLUS the model's own earlier predictions (raw kWh), so lags stay meaningful
    across the whole 168h run."""
    def lag_scaled(hours_back, col):
        key = ts - pd.Timedelta(hours=hours_back)
        if key in kwh_map:
            raw = kwh_map[key]
        else:
            raw = last_kwh
            counters["lag_fallbacks"] += 1
        return scalers[col].transform([[raw]])[0, 0]

    cal = compute_calendar_features(ts)
    temp, rad, hum = weather_row["temperature"], weather_row["shortwave_radiation"], weather_row["humidity"]
    holiday = get_holiday_flags([ts], country_code, state)[0]

    def sc(col, v):
        return scalers[col].transform([[v]])[0, 0]

    vals = {
        "kwh": next_kwh_scaled,
        "lag_1h": lag_scaled(1, "lag_1h"),
        "lag_24h": lag_scaled(24, "lag_24h"),
        "lag_48h": lag_scaled(48, "lag_48h"),
        "lag_168h": lag_scaled(168, "lag_168h"),
        "hour_sin": sc("hour_sin", cal["hour_sin"]), "hour_cos": sc("hour_cos", cal["hour_cos"]),
        "dow_sin": sc("dow_sin", cal["dow_sin"]), "dow_cos": sc("dow_cos", cal["dow_cos"]),
        "rolling_7d_mean": static["rolling_7d_mean"], "rolling_7d_std": static["rolling_7d_std"],
        "rolling_30d_mean": static["rolling_30d_mean"], "rolling_30d_std": static["rolling_30d_std"],
        "is_peak": sc("is_peak", is_peak_hour(ts.hour)),
        "season": sc("season", cal["season"]),
        "temperature": sc("temperature", temp),
        "shortwave_radiation": sc("shortwave_radiation", rad),
        "humidity": sc("humidity", hum),
        "controlled_load_kwh": sc("controlled_load_kwh", 0.0),
        "gross_generation_kwh": sc("gross_generation_kwh", 0.0),
        "net_generation_kwh": sc("net_generation_kwh", 0.0),
        "heating_degree": sc("heating_degree", max(0.0, 18.0 - temp)),
        "cooling_degree": sc("cooling_degree", max(0.0, temp - 18.0)),
        "is_public_holiday": sc("is_public_holiday", holiday),
    }
    return np.array([vals[c] for c in FEATURE_COLUMNS], dtype="float32")


def _fit_callbacks(has_val: bool):
    if not has_val:
        return []
    return [keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def run_live_forecast_all_horizons(user_csv_path: str, location: str,
                                    country_code: str = "AU", state: str = "NSW") -> dict:
    notes = []
    print("Loading CSV...")
    user_df = load_user_csv(user_csv_path)
    n_rows = len(user_df)
    min_rows = RECURSIVE_HORIZON + REC_LOOKBACK + 48
    if n_rows < min_rows:
        raise ValueError(f"Need at least {min_rows} hourly rows ({min_rows // 24} days); got {n_rows}.")

    print(f"Geocoding {location!r}...")
    geo = geocode_location(location)
    print(f"  -> {geo['resolved_name']} ({geo['lat']}, {geo['lon']})")

    first_ts, last_ts = user_df["timestamp"].min(), user_df["timestamp"].max()
    print(f"Fetching historical weather {first_ts.date()} to {last_ts.date()}...")
    hist_weather = fetch_weather(geo["lat"], geo["lon"], first_ts, last_ts)
    miss_hist = int(hist_weather.isna().any(axis=1).sum())
    if miss_hist:
        print(f"  {miss_hist} historical weather hours missing -- forward/back filled")
        notes.append(f"{miss_hist} historical weather hours were filled from neighbouring hours.")

    feature_df = build_live_feature_vector(user_df, hist_weather, country_code, state)
    scaled, scalers = scale_columns(feature_df)
    kwh_scaler = scalers["kwh"]
    last_timestamp = feature_df.index[-1]

    result = {
        "model": "bidirectional CNN-LSTM",
        "location": geo,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_period": {"start": str(first_ts), "end": str(last_ts), "hours": n_rows},
        "horizons": {},
    }

    # ---------------- 24h + 7-day: recursive with weather ----------------
    print("\n=== Recursive forecast (24h + 7-day), with weather ===")
    fc_start = last_timestamp + pd.Timedelta(hours=1)
    fc_end = last_timestamp + pd.Timedelta(hours=RECURSIVE_HORIZON)
    fc_weather = fetch_weather(geo["lat"], geo["lon"], fc_start, fc_end)

    # Impute any hours the sources could not provide: same hour-of-day mean of the last 14 days.
    recent = hist_weather.ffill().bfill().iloc[-14 * 24:]
    hod_mean = recent.groupby(recent.index.hour).mean()
    imputed = 0
    for ts in fc_weather.index[fc_weather.isna().any(axis=1)]:
        fc_weather.loc[ts] = hod_mean.loc[ts.hour]
        imputed += 1
    weather_source = "live/archive weather for the forecast window"
    if imputed:
        print(f"  WARNING: {imputed}/{RECURSIVE_HORIZON} forecast-window hours had no weather -- imputed from recent averages")
        notes.append(f"{imputed} of {RECURSIVE_HORIZON} forecast-window weather hours were unavailable and were "
                     f"imputed from the household's recent same-hour weather average.")
    if fc_end.normalize() < pd.Timestamp.now().normalize() - pd.Timedelta(days=ARCHIVE_LAG_DAYS):
        notes.append("The CSV ends in the past, so this is a BACKTEST-style run: the forecast window uses "
                     "archived actual weather, not a true forecast.")
        weather_source = "archived actual weather (backtest)"

    set_seeds(42)
    rec_model = keras.models.load_model(REC_MODEL_PATH, compile=False)
    rec_model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss="huber")
    X, y = create_sequences(scaled, REC_LOOKBACK)
    split = max(1, int(len(X) * 0.9)) if len(X) >= 10 else len(X)
    has_val = split < len(X)
    rec_model.fit(X[:split], y[:split],
                  validation_data=(X[split:], y[split:]) if has_val else None,
                  epochs=FINE_TUNE_EPOCHS, batch_size=16, callbacks=_fit_callbacks(has_val), verbose=0)

    static = {c: scaled[-1, FEATURE_COLUMNS.index(c)]
              for c in ["rolling_7d_mean", "rolling_7d_std", "rolling_30d_mean", "rolling_30d_std"]}
    kwh_map = {ts: float(v) for ts, v in zip(user_df["timestamp"], user_df["kwh"])}
    last_kwh = float(user_df["kwh"].iloc[-1])
    counters = {"lag_fallbacks": 0}

    window = scaled[-REC_LOOKBACK:, :].copy()
    preds_scaled, preds_kwh = [], []
    for step in range(RECURSIVE_HORIZON):
        ts = fc_start + pd.Timedelta(hours=step)
        p_scaled = float(rec_model.predict(window.reshape(1, REC_LOOKBACK, len(FEATURE_COLUMNS)), verbose=0)[0, 0])
        p_kwh = max(0.0, float(kwh_scaler.inverse_transform([[p_scaled]])[0, 0]))
        preds_scaled.append(p_scaled)
        preds_kwh.append(p_kwh)
        kwh_map[ts] = p_kwh
        # Row for ts: kwh = this prediction, lags read from kwh_map (history + earlier predictions).
        row = build_live_next_row(p_scaled, ts, kwh_map, last_kwh, fc_weather.loc[ts], scalers, static,
                                  country_code, state, counters)
        window = np.vstack([window[1:], row])

    rec_kwh = np.array(preds_kwh)
    rec_ts = pd.date_range(fc_start, periods=RECURSIVE_HORIZON, freq="h")
    result["horizons"]["24h"] = {
        "unit": "kWh per hour",
        "predictions": [{"timestamp": t.isoformat(), "predicted_kwh": round(float(v), 4)}
                        for t, v in zip(rec_ts[:24], rec_kwh[:24])],
        "total_kwh": round(float(rec_kwh[:24].sum()), 3),
        "method": f"recursive, {weather_source}",
        "validation": VALIDATION_EVIDENCE["24h"],
    }
    result["horizons"]["7day"] = {
        "unit": "kWh per day",
        "predictions": daily_totals(rec_ts, rec_kwh, 7),
        "total_kwh": round(float(rec_kwh.sum()), 3),
        "method": f"recursive, {weather_source}, summed per day",
        "validation": VALIDATION_EVIDENCE["7day"],
    }
    if counters["lag_fallbacks"]:
        notes.append(f"{counters['lag_fallbacks']} lag lookups hit a gap and used the last real value.")

    # ---------------- 1-month: direct multi-step, no weather ----------------
    print("\n=== Direct multi-step forecast (1-month), no weather ===")
    need = DM_LOOKBACK + MONTH_HORIZON + 48
    if not os.path.exists(MONTH_MODEL_PATH):
        result["horizons"]["1month"] = {"error": f"No 1-month model at {MONTH_MODEL_PATH}. "
                                                  f"Run bidirectional_direct_multistep_1month.py first."}
    elif len(scaled) < need:
        result["horizons"]["1month"] = {
            "error": f"Need at least {need} hourly feature rows (~{need // 24} days) to fine-tune the 1-month model; "
                     f"have {len(scaled)}. 24h and 7-day forecasts above are unaffected."}
    else:
        set_seeds(42)
        m_model = keras.models.load_model(MONTH_MODEL_PATH, compile=False)
        m_model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss="huber")
        Xm, ym = create_direct_multistep_sequences(scaled, DM_LOOKBACK, MONTH_HORIZON)
        msplit = max(1, int(len(Xm) * 0.9)) if len(Xm) >= 10 else len(Xm)
        m_has_val = msplit < len(Xm)
        m_model.fit(Xm[:msplit], ym[:msplit],
                    validation_data=(Xm[msplit:], ym[msplit:]) if m_has_val else None,
                    epochs=FINE_TUNE_EPOCHS, batch_size=16, callbacks=_fit_callbacks(m_has_val), verbose=0)
        last_window = scaled[-DM_LOOKBACK:, :].reshape(1, DM_LOOKBACK, len(FEATURE_COLUMNS))
        month_scaled = m_model.predict(last_window, verbose=0).flatten()
        month_kwh = np.clip(kwh_scaler.inverse_transform(month_scaled.reshape(-1, 1)).flatten(), 0, None)
        month_ts = pd.date_range(fc_start, periods=MONTH_HORIZON, freq="h")
        result["horizons"]["1month"] = {
            "unit": "kWh per day",
            "predictions": daily_totals(month_ts, month_kwh, MONTH_HORIZON // 24),
            "total_kwh": round(float(month_kwh.sum()), 3),
            "method": "direct multi-step, no weather, summed per day",
            "validation": VALIDATION_EVIDENCE["1month"],
        }

    result["notes"] = notes
    result["limitations"] = [
        "controlled_load_kwh / gross_generation_kwh / net_generation_kwh are set to 0 -- a new household's raw data "
        "has no such metering breakdown.",
        "is_peak uses a fixed definition (7-9am, 5-9pm); public holidays use "
        f"{country_code}/{state} -- override with --country/--state if wrong.",
        "The 1-month forecast uses no future weather and tracks the level, not hour-by-hour peaks.",
        "Accuracy figures come from cross-validation on ONE held-out-in-time household (8162493). "
        "A genuinely new household has not been validated -- treat as provisional.",
    ]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-csv", required=True, help="CSV with columns: timestamp, kwh")
    parser.add_argument("--location", required=True, help="e.g. 'Newcastle, Australia'")
    parser.add_argument("--country", default="AU")
    parser.add_argument("--state", default="NSW")
    parser.add_argument("--out", default="live_forecast_bidirectional.json")
    args = parser.parse_args()

    result = run_live_forecast_all_horizons(args.user_csv, args.location, args.country, args.state)

    print(f"\nLocation: {result['location']['resolved_name']}")
    for name, data in result["horizons"].items():
        if "error" in data:
            print(f"\n{name}: ERROR - {data['error']}")
            continue
        print(f"\n{name}: {len(data['predictions'])} predictions ({data['unit']}), total {data['total_kwh']} kWh -- {data['method']}")
    for n in result.get("notes", []):
        print(f"  note: {n}")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved full result to {args.out}")


if __name__ == "__main__":
    main()
