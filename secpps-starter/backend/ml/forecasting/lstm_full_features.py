"""
LSTM using the FULL proposal-specified feature vector F(t):
  - Mishael's engineered features (lag_1h/24h/48h/168h, hour/dow cyclical,
    season, 7d/30d rolling mean+std, is_peak) from feature_matrix.csv
  - Your own weather data (temperature, shortwave_radiation) joined in

See conversation for the full reasoning table on how each feature type
is handled during the 24-hour recursive forecast -- summarized in code
comments below at the point each is used.

STILL MISSING (not fabricated, flagged honestly): public holiday
indicators are not present in either data source and are not included
here -- a genuine gap against the full proposal spec, not silently
patched over.
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

# Column 0 MUST be the prediction target (kwh) -- code below assumes this.
FEATURE_COLUMNS = [
    "kwh",
    "lag_1h", "lag_24h", "lag_48h", "lag_168h",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "rolling_7d_mean", "rolling_7d_std", "rolling_30d_mean", "rolling_30d_std",
    "is_peak", "season",
    "temperature", "shortwave_radiation",
]


def load_features_with_weather(engine, household_id: int) -> pd.DataFrame:
    feat_df = pd.read_csv(FEATURE_MATRIX_PATH)
    feat_df["timestamp"] = pd.to_datetime(feat_df["timestamp"])
    feat_df = feat_df[feat_df["household_id"] == household_id].sort_values("timestamp")
    if feat_df.empty:
        raise ValueError(f"No rows for household_id={household_id} in {FEATURE_MATRIX_PATH}")

    weather_query = """
        SELECT reading_timestamp AS timestamp, temperature, shortwave_radiation
        FROM weather_readings
        WHERE household_id = %(household_id)s
    """
    weather_df = pd.read_sql(weather_query, engine, params={"household_id": household_id})
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])

    merged = feat_df.merge(weather_df, on="timestamp", how="inner")
    merged = merged.set_index("timestamp").sort_index()

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
        y.append(data[i + lookback, 0])  # kwh is column 0
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
    """hour/dow cyclical + season, computed fresh for any real timestamp --
    pure calendar math, zero dependency on the target series, so this is
    always legitimate to compute for a real future (or held-out) date."""
    hour_angle = 2 * np.pi * ts.hour / 24
    dow_angle = 2 * np.pi * ts.dayofweek / 7
    # Southern Hemisphere season mapping (consistent with earlier synthetic data)
    month = ts.month
    if month in (12, 1, 2):
        season = 0  # summer
    elif month in (3, 4, 5):
        season = 1  # autumn
    elif month in (6, 7, 8):
        season = 2  # winter
    else:
        season = 3  # spring
    return {
        "hour_sin": np.sin(hour_angle), "hour_cos": np.cos(hour_angle),
        "dow_sin": np.sin(dow_angle), "dow_cos": np.cos(dow_angle),
        "season": season,
    }


def generate_full_feature_forecast(
    engine, household_id: int, lookback: int = LOOKBACK, units: int = 32, dropout: float = 0.2,
) -> dict:
    set_seeds()

    df = load_features_with_weather(engine, household_id)
    if len(df) < HORIZON + lookback + 100:
        raise ValueError(f"Not enough data for household {household_id}: {len(df)} rows")

    train_df = df.iloc[:-HORIZON]
    test_df = df.iloc[-HORIZON:]  # real actual values -- ground truth AND source for lag_24h/48h/168h lookups

    scaled_train, scalers = scale_columns(train_df)
    X, y = create_sequences(scaled_train, lookback)

    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    model = build_model(n_features=len(FEATURE_COLUMNS), lookback=lookback, units=units, dropout=dropout)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                         epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)

    # ---- Recursive 24-hour forecast, feature-by-feature handling ----
    kwh_scaler = scalers["kwh"]
    window = scaled_train[-lookback:, :].copy()

    # Held-constant rolling/peak values, from the last real training row
    # (documented simplification -- see conversation for reasoning)
    last_rolling_and_peak = {
        col: scaled_train[-1, FEATURE_COLUMNS.index(col)]
        for col in ["rolling_7d_mean", "rolling_7d_std", "rolling_30d_mean", "rolling_30d_std", "is_peak"]
    }

    scaled_predictions = []

    for step in range(HORIZON):
        ts = test_df.index[step]

        x_input = window.reshape((1, lookback, len(FEATURE_COLUMNS)))
        next_kwh_scaled = model.predict(x_input, verbose=0)[0, 0]
        scaled_predictions.append(next_kwh_scaled)

        calendar = compute_calendar_features(ts)

        # lag_1h: recursive -- the value just predicted becomes "1 hour ago" for the NEXT step
        lag_1h_scaled = next_kwh_scaled

        # lag_24h/48h/168h: always >=24h in the past for a 24h horizon,
        # so always available as REAL known values -- no leakage
        def real_lag_scaled(hours_back, col):
            target_time = ts - pd.Timedelta(hours=hours_back)
            if target_time in train_df.index:
                raw_val = train_df.loc[target_time, "kwh"]
            else:
                raw_val = train_df["kwh"].iloc[-1]  # fallback, shouldn't normally trigger
            return scalers[col].transform([[raw_val]])[0, 0]

        lag_24h_scaled = real_lag_scaled(24, "lag_24h")
        lag_48h_scaled = real_lag_scaled(48, "lag_48h")
        lag_168h_scaled = real_lag_scaled(168, "lag_168h")

        # Real actual test-period weather (same caveat as every other weather script)
        temp_scaled = scalers["temperature"].transform([[test_df.iloc[step]["temperature"]]])[0, 0]
        radiation_scaled = scalers["shortwave_radiation"].transform([[test_df.iloc[step]["shortwave_radiation"]]])[0, 0]

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
            temp_scaled, radiation_scaled,
        ])
        window = np.vstack([window[1:], next_row])

    predicted_kwh = kwh_scaler.inverse_transform(np.array(scaled_predictions).reshape(-1, 1)).flatten()
    metrics = compute_metrics(test_df["kwh"].values, predicted_kwh)

    predictions = [
        {"timestamp": ts.isoformat(), "predicted_kwh": round(float(v), 4)}
        for ts, v in zip(test_df.index, predicted_kwh)
    ]

    return {
        "household_id": household_id,
        "model": "LSTM (full feature vector: lag+cyclical+rolling+peak+season+weather)",
        "feature_columns": FEATURE_COLUMNS,
        "training": {
            "epochs_run": len(history.history["loss"]),
            "final_train_loss": round(float(history.history["loss"][-1]), 6),
            "final_val_loss": round(float(history.history["val_loss"][-1]), 6),
        },
        "predictions": predictions,
        "metrics": metrics,
        "known_gaps": ["public holiday indicators not available in current data sources"],
        "simplifications": [
            "rolling 7d/30d mean+std and is_peak held constant at last known training value during the 24h forecast",
            "weather uses real actual test-period values, not a forecast -- valid for backtesting only",
        ],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--household-id", type=int, required=True)
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)
    result = generate_full_feature_forecast(engine, args.household_id)

    print(f"Household {result['household_id']}")
    print(f"Training: {result['training']}")
    print(f"Metrics: {result['metrics']}")
    print(f"First 3 predictions: {result['predictions'][:3]}")

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"lstm_full_features_{args.household_id}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {path}")
