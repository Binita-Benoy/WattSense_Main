"""
Shared foundation for the final combined model: feature loading, scaling,
sequence-building, and the recursive-forecast row builder. Every other
file in this folder imports from here, so there is exactly ONE place
that defines the feature vector -- avoids the column-mismatch bugs that
hit earlier versions of this pipeline when the same logic was
duplicated across files.

24-column feature vector:
  kwh (target) + 4 lags + 4 cyclical + 4 rolling + is_peak + season
  + temperature + shortwave_radiation + humidity
  + controlled_load_kwh + gross_generation_kwh + net_generation_kwh
  + heating_degree + cooling_degree + is_public_holiday
"""
import os

import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.preprocessing import RobustScaler

FEATURE_MATRIX_PATH = os.getenv(
    "FEATURE_MATRIX_PATH",
    os.path.join(os.path.dirname(__file__), "..", "feature_matrix.csv"),
)

HORIZON = 24

FEATURE_COLUMNS = [
    "kwh",
    "lag_1h", "lag_24h", "lag_48h", "lag_168h",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "rolling_7d_mean", "rolling_7d_std", "rolling_30d_mean", "rolling_30d_std",
    "is_peak", "season",
    "temperature", "shortwave_radiation", "humidity",
    "controlled_load_kwh", "gross_generation_kwh", "net_generation_kwh",
    "heating_degree", "cooling_degree", "is_public_holiday",
]

WEATHER_COLS = ["temperature", "shortwave_radiation", "humidity"]
GENERATION_COLS = ["controlled_load_kwh", "gross_generation_kwh", "net_generation_kwh"]
DEGREE_HOLIDAY_COLS = ["heating_degree", "cooling_degree", "is_public_holiday"]


_FEATURE_CSV_CACHE = None


def _get_feature_csv() -> pd.DataFrame:
    """Reads the feature matrix CSV ONCE per process, cached in memory --
    without this, calling load_features() for N households means reading
    the same large CSV from disk N separate times, which is genuinely
    heavy, sustained disk I/O and CPU work for no benefit."""
    global _FEATURE_CSV_CACHE
    if _FEATURE_CSV_CACHE is None:
        _FEATURE_CSV_CACHE = pd.read_csv(FEATURE_MATRIX_PATH)
        _FEATURE_CSV_CACHE["timestamp"] = pd.to_datetime(_FEATURE_CSV_CACHE["timestamp"])
    return _FEATURE_CSV_CACHE


def load_features(engine, household_id: int) -> pd.DataFrame:
    feat_df = _get_feature_csv()
    feat_df = feat_df[feat_df["household_id"] == household_id].sort_values("timestamp")
    if feat_df.empty:
        raise ValueError(f"No rows for household_id={household_id} in {FEATURE_MATRIX_PATH}")

    if all(c in feat_df.columns for c in WEATHER_COLS):
        merged = feat_df.set_index("timestamp").sort_index()
    else:
        q = """
            SELECT reading_timestamp AS timestamp, temperature, shortwave_radiation, humidity
            FROM weather_readings WHERE household_id = %(household_id)s
        """
        weather_df = pd.read_sql(q, engine, params={"household_id": household_id})
        weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
        merged = feat_df.merge(weather_df, on="timestamp", how="inner").set_index("timestamp").sort_index()

    if not all(c in merged.columns for c in GENERATION_COLS):
        q = """
            SELECT reading_timestamp AS timestamp, controlled_load_kwh, gross_generation_kwh, net_generation_kwh
            FROM energy_readings WHERE household_id = %(household_id)s
        """
        extra_df = pd.read_sql(q, engine, params={"household_id": household_id})
        extra_df["timestamp"] = pd.to_datetime(extra_df["timestamp"])
        merged = merged.reset_index().merge(extra_df, on="timestamp", how="left").set_index("timestamp")
    for col in GENERATION_COLS:
        merged[col] = merged[col].fillna(0)

    if not all(c in merged.columns for c in DEGREE_HOLIDAY_COLS):
        q = """
            SELECT reading_timestamp AS timestamp, is_public_holiday
            FROM energy_readings WHERE household_id = %(household_id)s
        """
        holiday_df = pd.read_sql(q, engine, params={"household_id": household_id})
        holiday_df["timestamp"] = pd.to_datetime(holiday_df["timestamp"])
        merged = merged.reset_index().merge(holiday_df, on="timestamp", how="left").set_index("timestamp")
        merged["is_public_holiday"] = merged["is_public_holiday"].fillna(0)
        base_temp = 18.0
        if "heating_degree" not in merged.columns:
            merged["heating_degree"] = (base_temp - merged["temperature"]).clip(lower=0)
        if "cooling_degree" not in merged.columns:
            merged["cooling_degree"] = (merged["temperature"] - base_temp).clip(lower=0)

    missing = [c for c in FEATURE_COLUMNS if c not in merged.columns]
    if missing:
        raise ValueError(f"Expected columns missing after merge: {missing}")

    return merged[FEATURE_COLUMNS].dropna()


def scale_columns(df: pd.DataFrame):
    scalers = {}
    scaled = np.zeros_like(df.values, dtype=np.float32)  # float32, not float64 -- halves memory for large sequences
    for i, col in enumerate(FEATURE_COLUMNS):
        scaler = RobustScaler()
        scaled[:, i] = scaler.fit_transform(df[[col]].values).flatten().astype(np.float32)
        scalers[col] = scaler
    return scaled, scalers


def create_sequences(data: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback, :])
        y.append(data[i + lookback, 0])
    return np.array(X), np.array(y)


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"mae": round(float(mae), 4), "rmse": round(float(rmse), 4), "mape": round(float(mape), 2)}


def set_seeds(seed: int = 42):
    import random
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compute_calendar_features(ts: pd.Timestamp) -> dict:
    hour_angle = 2 * np.pi * ts.hour / 24
    dow_angle = 2 * np.pi * ts.dayofweek / 7
    month = ts.month
    season = 0 if month in (12, 1, 2) else 1 if month in (3, 4, 5) else 2 if month in (6, 7, 8) else 3
    return {
        "hour_sin": np.sin(hour_angle), "hour_cos": np.cos(hour_angle),
        "dow_sin": np.sin(dow_angle), "dow_cos": np.cos(dow_angle),
        "season": season,
    }


def build_next_row(next_kwh_scaled, ts, train_df, test_df, step, scalers, last_rolling_and_peak):
    calendar = compute_calendar_features(ts)
    lag_1h_scaled = next_kwh_scaled

    def real_lag_scaled(hours_back, col):
        target_time = ts - pd.Timedelta(hours=hours_back)
        raw_val = train_df.loc[target_time, "kwh"] if target_time in train_df.index else train_df["kwh"].iloc[-1]
        return scalers[col].transform([[raw_val]])[0, 0]

    lag_24h_scaled = real_lag_scaled(24, "lag_24h")
    lag_48h_scaled = real_lag_scaled(48, "lag_48h")
    lag_168h_scaled = real_lag_scaled(168, "lag_168h")

    def real_val_scaled(col):
        return scalers[col].transform([[test_df.iloc[step][col]]])[0, 0]

    return np.array([
        next_kwh_scaled, lag_1h_scaled, lag_24h_scaled, lag_48h_scaled, lag_168h_scaled,
        scalers["hour_sin"].transform([[calendar["hour_sin"]]])[0, 0],
        scalers["hour_cos"].transform([[calendar["hour_cos"]]])[0, 0],
        scalers["dow_sin"].transform([[calendar["dow_sin"]]])[0, 0],
        scalers["dow_cos"].transform([[calendar["dow_cos"]]])[0, 0],
        last_rolling_and_peak["rolling_7d_mean"], last_rolling_and_peak["rolling_7d_std"],
        last_rolling_and_peak["rolling_30d_mean"], last_rolling_and_peak["rolling_30d_std"],
        last_rolling_and_peak["is_peak"],
        scalers["season"].transform([[calendar["season"]]])[0, 0],
        real_val_scaled("temperature"), real_val_scaled("shortwave_radiation"), real_val_scaled("humidity"),
        real_val_scaled("controlled_load_kwh"), real_val_scaled("gross_generation_kwh"), real_val_scaled("net_generation_kwh"),
        real_val_scaled("heating_degree"), real_val_scaled("cooling_degree"), real_val_scaled("is_public_holiday"),
    ])


def discover_real_households(engine, min_hours: int = 2000, limit: int = None) -> list:
    query = """
        SELECT h.household_id, count(DISTINCT e.reading_timestamp) AS n_hours
        FROM households h
        JOIN energy_readings e ON e.household_id = h.household_id
        WHERE h.source NOT IN ('synthetic', 'homec_kaggle', 'demo_house')
        GROUP BY h.household_id
        HAVING count(DISTINCT e.reading_timestamp) >= :min_hours
        ORDER BY h.household_id
    """
    with engine.connect() as conn:
        result = conn.execute(text(query), {"min_hours": min_hours})
        ids = [row[0] for row in result]
    return ids[:limit] if limit else ids
