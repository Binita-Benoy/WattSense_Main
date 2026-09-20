"""
LSTM using the FULL proposal-specified feature vector F(t), now extended
with SGSC's own generation/controlled-load columns:
  - Mishael's engineered features (lag/cyclical/rolling/peak/season)
  - Weather: temperature, shortwave_radiation, humidity
  - SGSC's own controlled_load_kwh, gross_generation_kwh, net_generation_kwh
    -- previously unused; households with solar (has_generation) may
    behave very differently, and this signal was sitting unused.

STILL MISSING (flagged honestly, not silently patched): public holiday
indicators are not available in current data sources.

Refactored: the core train+recursive-forecast logic now lives in
_train_and_forecast(), taking explicit train_df/test_df -- this lets
lstm_cv.py reuse it directly for rolling-origin cross-validation folds,
instead of duplicating the whole recursive loop a second time.
"""
import json
import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import RobustScaler

from tensorflow import keras
from tensorflow.keras import layers

from lstm_forecast import set_seeds, compute_metrics

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)
FEATURE_MATRIX_PATH = os.getenv(
    "FEATURE_MATRIX_PATH",
    os.path.join(os.path.dirname(__file__), "feature_matrix.csv"),
)

LOOKBACK = 24
HORIZON = 24

FEATURE_COLUMNS = [
    "kwh",
    "lag_1h", "lag_24h", "lag_48h", "lag_168h",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "rolling_7d_mean", "rolling_7d_std", "rolling_30d_mean", "rolling_30d_std",
    "is_peak", "season",
    "temperature", "shortwave_radiation", "humidity",
    "controlled_load_kwh", "gross_generation_kwh", "net_generation_kwh",
]

WEATHER_COLS = ["temperature", "shortwave_radiation", "humidity"]
GENERATION_COLS = ["controlled_load_kwh", "gross_generation_kwh", "net_generation_kwh"]


def load_features_with_weather(engine, household_id: int) -> pd.DataFrame:
    feat_df = pd.read_csv(FEATURE_MATRIX_PATH)
    feat_df["timestamp"] = pd.to_datetime(feat_df["timestamp"])
    feat_df = feat_df[feat_df["household_id"] == household_id].sort_values("timestamp")
    if feat_df.empty:
        raise ValueError(f"No rows for household_id={household_id} in {FEATURE_MATRIX_PATH}")

    # Weather: use directly if Mishael's pipeline already merged it in,
    # otherwise pull from weather_readings ourselves.
    if all(c in feat_df.columns for c in WEATHER_COLS):
        merged = feat_df.set_index("timestamp").sort_index()
    else:
        weather_query = """
            SELECT reading_timestamp AS timestamp, temperature, shortwave_radiation, humidity
            FROM weather_readings WHERE household_id = %(household_id)s
        """
        weather_df = pd.read_sql(weather_query, engine, params={"household_id": household_id})
        weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
        merged = feat_df.merge(weather_df, on="timestamp", how="inner").set_index("timestamp").sort_index()

    # Generation/controlled-load: same "use if present, else fetch" pattern.
    if not all(c in merged.columns for c in GENERATION_COLS):
        energy_query = """
            SELECT reading_timestamp AS timestamp, controlled_load_kwh, gross_generation_kwh, net_generation_kwh
            FROM energy_readings WHERE household_id = %(household_id)s
        """
        extra_df = pd.read_sql(energy_query, engine, params={"household_id": household_id})
        extra_df["timestamp"] = pd.to_datetime(extra_df["timestamp"])
        merged = merged.reset_index().merge(extra_df, on="timestamp", how="left").set_index("timestamp")

    # Households without solar/controlled-load legitimately have NULL here
    # -- treat as 0 (no generation/controlled load), not as missing data
    # to be dropped.
    for col in GENERATION_COLS:
        merged[col] = merged[col].fillna(0)

    missing = [c for c in FEATURE_COLUMNS if c not in merged.columns]
    if missing:
        raise ValueError(f"Expected columns missing after merge: {missing}")

    return merged[FEATURE_COLUMNS].dropna()


def scale_columns(df: pd.DataFrame):
    scalers = {}
    scaled = np.zeros_like(df.values, dtype=float)
    for i, col in enumerate(FEATURE_COLUMNS):
        scaler = RobustScaler()
        scaled[:, i] = scaler.fit_transform(df[[col]].values).flatten()
        scalers[col] = scaler
    return scaled, scalers


def create_sequences(data: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback, :])
        y.append(data[i + lookback, 0])
    return np.array(X), np.array(y)


def build_model(n_features: int, lookback: int = LOOKBACK, units: int = 32, dropout: float = 0.2):
    model = keras.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.LSTM(units),
        layers.Dropout(dropout),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


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


def _train_and_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame,
                         lookback: int = LOOKBACK, units: int = 32, dropout: float = 0.2,
                         horizon: int = HORIZON) -> dict:
    """
    Core reusable logic: train on train_df, recursively forecast `horizon`
    steps, evaluate against test_df's real values. Used both by the
    single-split entry point below AND by lstm_cv.py's fold loop.
    """
    set_seeds()

    scaled_train, scalers = scale_columns(train_df)
    X, y = create_sequences(scaled_train, lookback)

    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    model = build_model(n_features=len(FEATURE_COLUMNS), lookback=lookback, units=units, dropout=dropout)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                         epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)

    kwh_scaler = scalers["kwh"]
    window = scaled_train[-lookback:, :].copy()

    last_rolling_and_peak = {
        col: scaled_train[-1, FEATURE_COLUMNS.index(col)]
        for col in ["rolling_7d_mean", "rolling_7d_std", "rolling_30d_mean", "rolling_30d_std", "is_peak"]
    }

    scaled_predictions = []
    for step in range(horizon):
        ts = test_df.index[step]

        x_input = window.reshape((1, lookback, len(FEATURE_COLUMNS)))
        next_kwh_scaled = model.predict(x_input, verbose=0)[0, 0]
        scaled_predictions.append(next_kwh_scaled)

        calendar = compute_calendar_features(ts)
        lag_1h_scaled = next_kwh_scaled

        def real_lag_scaled(hours_back, col):
            target_time = ts - pd.Timedelta(hours=hours_back)
            raw_val = train_df.loc[target_time, "kwh"] if target_time in train_df.index else train_df["kwh"].iloc[-1]
            return scalers[col].transform([[raw_val]])[0, 0]

        lag_24h_scaled = real_lag_scaled(24, "lag_24h")
        lag_48h_scaled = real_lag_scaled(48, "lag_48h")
        lag_168h_scaled = real_lag_scaled(168, "lag_168h")

        temp_scaled = scalers["temperature"].transform([[test_df.iloc[step]["temperature"]]])[0, 0]
        radiation_scaled = scalers["shortwave_radiation"].transform([[test_df.iloc[step]["shortwave_radiation"]]])[0, 0]
        humidity_scaled = scalers["humidity"].transform([[test_df.iloc[step]["humidity"]]])[0, 0]
        controlled_load_scaled = scalers["controlled_load_kwh"].transform([[test_df.iloc[step]["controlled_load_kwh"]]])[0, 0]
        gross_gen_scaled = scalers["gross_generation_kwh"].transform([[test_df.iloc[step]["gross_generation_kwh"]]])[0, 0]
        net_gen_scaled = scalers["net_generation_kwh"].transform([[test_df.iloc[step]["net_generation_kwh"]]])[0, 0]

        next_row = np.array([
            next_kwh_scaled, lag_1h_scaled, lag_24h_scaled, lag_48h_scaled, lag_168h_scaled,
            scalers["hour_sin"].transform([[calendar["hour_sin"]]])[0, 0],
            scalers["hour_cos"].transform([[calendar["hour_cos"]]])[0, 0],
            scalers["dow_sin"].transform([[calendar["dow_sin"]]])[0, 0],
            scalers["dow_cos"].transform([[calendar["dow_cos"]]])[0, 0],
            last_rolling_and_peak["rolling_7d_mean"], last_rolling_and_peak["rolling_7d_std"],
            last_rolling_and_peak["rolling_30d_mean"], last_rolling_and_peak["rolling_30d_std"],
            last_rolling_and_peak["is_peak"],
            scalers["season"].transform([[calendar["season"]]])[0, 0],
            temp_scaled, radiation_scaled, humidity_scaled,
            controlled_load_scaled, gross_gen_scaled, net_gen_scaled,
        ])
        window = np.vstack([window[1:], next_row])

    predicted_kwh = kwh_scaler.inverse_transform(np.array(scaled_predictions).reshape(-1, 1)).flatten()
    metrics = compute_metrics(test_df["kwh"].values[:horizon], predicted_kwh)

    predictions = [
        {"timestamp": ts.isoformat(), "predicted_kwh": round(float(v), 4)}
        for ts, v in zip(test_df.index[:horizon], predicted_kwh)
    ]

    return {
        "metrics": metrics,
        "predictions": predictions,
        "training": {
            "epochs_run": len(history.history["loss"]),
            "final_val_loss": round(float(history.history["val_loss"][-1]), 6),
        },
    }


def generate_full_feature_forecast(
    engine, household_id: int, lookback: int = LOOKBACK, units: int = 32, dropout: float = 0.2,
) -> dict:
    df = load_features_with_weather(engine, household_id)
    if len(df) < HORIZON + lookback + 100:
        raise ValueError(f"Not enough data for household {household_id}: {len(df)} rows")

    train_df = df.iloc[:-HORIZON]
    test_df = df.iloc[-HORIZON:]

    core = _train_and_forecast(train_df, test_df, lookback, units, dropout, HORIZON)

    return {
        "household_id": household_id,
        "model": "LSTM (full feature vector incl. generation/controlled-load)",
        "feature_columns": FEATURE_COLUMNS,
        "lookback": lookback, "units": units, "dropout": dropout,
        "training": core["training"],
        "predictions": core["predictions"],
        "metrics": core["metrics"],
        "known_gaps": ["public holiday indicators not available in current data sources"],
        "simplifications": [
            "rolling 7d/30d mean+std and is_peak held constant at last known training value during the 24h forecast",
            "weather and generation/controlled-load use real actual test-period values, not forecasts -- valid for backtesting only",
        ],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--household-id", type=int, required=True)
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--units", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)
    result = generate_full_feature_forecast(
        engine, args.household_id, lookback=args.lookback, units=args.units, dropout=args.dropout
    )

    print(f"Household {result['household_id']} (lookback={result['lookback']}, units={result['units']})")
    print(f"Training: {result['training']}")
    print(f"Metrics: {result['metrics']}")
    print(f"First 3 predictions: {result['predictions'][:3]}")

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"lstm_full_features_{args.household_id}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {path}")