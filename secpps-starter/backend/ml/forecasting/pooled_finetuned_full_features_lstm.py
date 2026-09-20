"""
Pooled + fine-tuned LSTM -- your proposal-aligned production model,
now with the full 21-column feature vector (lag/cyclical/rolling/peak/
season/weather/humidity/generation/controlled-load), matching
lstm_full_features.py exactly.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from tensorflow import keras

from lstm_forecast import set_seeds, compute_metrics
from lstm_full_features import (
    load_features_with_weather, scale_columns, create_sequences, build_model,
    compute_calendar_features, FEATURE_COLUMNS, LOOKBACK, HORIZON,
)

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pooled_full_features_lstm.keras")
METADATA_PATH = os.path.join(MODEL_DIR, "pooled_full_features_metadata.json")

FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 0.0001


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


def _build_next_row(next_kwh_scaled, ts, train_df, test_df, step, scalers, last_rolling_and_peak):
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
        temp_scaled, radiation_scaled, humidity_scaled,
        controlled_load_scaled, gross_gen_scaled, net_gen_scaled,
    ])


def train_pooled_base_model(engine, household_ids: list, lookback: int = LOOKBACK, epochs: int = 50):
    set_seeds()

    print(f"Preparing full-feature sequences from {len(household_ids)} households: {household_ids}")
    X_pool, y_pool = [], []
    for hid in household_ids:
        df = load_features_with_weather(engine, hid)
        train_df = df.iloc[:-HORIZON]

        scaled_train, _ = scale_columns(train_df)
        X, y = create_sequences(scaled_train, lookback)
        X_pool.append(X)
        y_pool.append(y)

    X_pool = np.concatenate(X_pool, axis=0)
    y_pool = np.concatenate(y_pool, axis=0)
    print(f"Total pooled sequences: {len(X_pool)}, features per timestep: {X_pool.shape[2]}")

    split_idx = int(len(X_pool) * 0.9)
    X_train, X_val = X_pool[:split_idx], X_pool[split_idx:]
    y_train, y_val = y_pool[:split_idx], y_pool[split_idx:]

    model = build_model(n_features=len(FEATURE_COLUMNS), lookback=lookback)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                         epochs=epochs, batch_size=32, callbacks=[early_stop], verbose=0)
    print(f"Base training complete after {len(history.history['loss'])} epochs, "
          f"final val loss: {history.history['val_loss'][-1]:.6f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump({
            "trained_on_households": household_ids,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "lookback": lookback,
            "features": FEATURE_COLUMNS,
        }, f, indent=2)
    print(f"Pooled base model saved to {MODEL_PATH}")
    return model


def fine_tune_and_forecast(engine, household_id: int, lookback: int = LOOKBACK) -> dict:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("No pooled base model found -- run --mode train first")

    set_seeds()
    model = keras.models.load_model(MODEL_PATH)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss="mse")

    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    was_in_training_pool = household_id in metadata["trained_on_households"]

    df = load_features_with_weather(engine, household_id)
    train_df = df.iloc[:-HORIZON]
    test_df = df.iloc[-HORIZON:]

    scaled_train, scalers = scale_columns(train_df)
    X, y = create_sequences(scaled_train, lookback)

    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=FINE_TUNE_EPOCHS, batch_size=32, callbacks=[early_stop], verbose=0)

    kwh_scaler = scalers["kwh"]
    window = scaled_train[-lookback:, :].copy()

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

        next_row = _build_next_row(next_kwh_scaled, ts, train_df, test_df, step, scalers, last_rolling_and_peak)
        window = np.vstack([window[1:], next_row])

    predicted_kwh = kwh_scaler.inverse_transform(np.array(scaled_predictions).reshape(-1, 1)).flatten()
    metrics = compute_metrics(test_df["kwh"].values, predicted_kwh)

    predictions = [
        {"timestamp": ts.isoformat(), "predicted_kwh": round(float(v), 4)}
        for ts, v in zip(test_df.index, predicted_kwh)
    ]

    return {
        "household_id": household_id,
        "model": "Pooled + fine-tuned LSTM (full feature vector incl. generation/controlled-load)",
        "was_in_training_pool": was_in_training_pool,
        "trained_on_households": metadata["trained_on_households"],
        "predictions": predictions,
        "metrics": metrics,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "forecast"], required=True)
    parser.add_argument("--household-ids", type=str, help="Comma-separated, for --mode train")
    parser.add_argument("--household-id", type=int, help="Single household, for --mode forecast")
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)

    if args.mode == "train":
        if args.household_ids:
            household_ids = [int(x) for x in args.household_ids.split(",")]
        else:
            household_ids = discover_real_households(engine)
            print(f"No --household-ids given, auto-discovered: {household_ids}")
        train_pooled_base_model(engine, household_ids)

    elif args.mode == "forecast":
        if not args.household_id:
            raise ValueError("--household-id required for --mode forecast")
        result = fine_tune_and_forecast(engine, args.household_id)

        print(f"\nHousehold {result['household_id']}")
        print(f"Was in base training pool: {result['was_in_training_pool']}")
        print(f"Metrics: {result['metrics']}")

        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, f"pooled_finetuned_full_features_{args.household_id}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {path}")