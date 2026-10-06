"""
Retrains the global pooled sequential Conv1D-LSTM as a DIRECT multi-step
model -- predicts the entire horizon in one forward pass from the 24-hour
lookback window, rather than recursively stepping hour by hour.

WHY: recursive forecasting needs real future weather at every step, which
only exists (as an actual forecast, not history) for ~16 days out via
Open-Meteo. Direct multi-step sidesteps this entirely -- it never takes
future weather as input at all, only the past lookback window. The
honest cost: it cannot react to a specific future heatwave or cold
snap -- it forecasts from learned historical patterns only. This is a
real limitation, not a bug.

Trains one separate model per horizon (24h, 168h/7-day, 720h/1-month).
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import create_engine
from tensorflow import keras
from tensorflow.keras import layers

from features import (
    load_features, scale_columns, set_seeds, FEATURE_COLUMNS, discover_real_households,
)

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)

LOOKBACK = 24
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

HORIZONS = {"24h": 24, "7day": 168, "1month": 720}


def create_direct_multistep_sequences(data: np.ndarray, lookback: int, horizon: int):
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        X.append(data[i:i + lookback, :])
        y.append(data[i + lookback: i + lookback + horizon, 0])
    return np.array(X), np.array(y)


def build_direct_multistep_model(n_features: int, horizon: int, units: int = 32, filters: int = 32, dropout: float = 0.2):
    model = keras.Sequential([
        layers.Input(shape=(LOOKBACK, n_features)),
        layers.Conv1D(filters=filters, kernel_size=3, activation="relu", padding="causal"),
        layers.LSTM(units),
        layers.Dropout(dropout),
        layers.Dense(horizon),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_global_model_for_horizon(engine, household_ids: list, horizon_name: str, units: int = 32):
    horizon = HORIZONS[horizon_name]
    set_seeds()

    X_train_pool, y_train_pool = [], []
    val_per_household = []
    for hid in household_ids:
        df = load_features(engine, hid)
        scaled, scalers = scale_columns(df)
        X, y = create_direct_multistep_sequences(scaled, LOOKBACK, horizon)
        if len(X) == 0:
            continue
        split_idx = int(len(X) * 0.9)
        X_train_pool.append(X[:split_idx])
        y_train_pool.append(y[:split_idx])
        val_per_household.append((X[split_idx:], y[split_idx:], scalers["kwh"]))

    X_train = np.concatenate(X_train_pool)
    y_train = np.concatenate(y_train_pool)
    X_val = np.concatenate([v[0] for v in val_per_household if len(v[0]) > 0])
    y_val = np.concatenate([v[1] for v in val_per_household if len(v[0]) > 0])

    model = build_direct_multistep_model(len(FEATURE_COLUMNS), horizon, units=units)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    print(f"Training direct multi-step model for horizon={horizon_name} ({horizon}h), "
          f"{len(X_train)} train sequences, {len(X_val)} validation...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=50, batch_size=32, callbacks=[early_stop], verbose=2)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"global_direct_multistep_{horizon_name}.keras")
    model.save(model_path)
    metadata_path = os.path.join(MODEL_DIR, f"global_direct_multistep_{horizon_name}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({"horizon_name": horizon_name, "horizon_hours": horizon, "units": units,
                    "trained_on_households": household_ids, "lookback": LOOKBACK,
                    "features": FEATURE_COLUMNS,
                    "trained_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)
    print(f"Saved to {model_path}")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", choices=list(HORIZONS.keys()), required=True)
    parser.add_argument("--units", type=int, default=32)
    parser.add_argument("--pool-limit", type=int, default=None,
                         help="Limit households pooled for training -- use if the full 30-household "
                              "pool hits Docker's memory ceiling (this happened for the 1month horizon)")
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)
    household_ids = discover_real_households(engine, limit=args.pool_limit)
    print(f"Pooling {len(household_ids)} households"
          f"{f' (pool-limit={args.pool_limit})' if args.pool_limit else ''}: {household_ids}\n")
    train_global_model_for_horizon(engine, household_ids, args.horizon, units=args.units)
