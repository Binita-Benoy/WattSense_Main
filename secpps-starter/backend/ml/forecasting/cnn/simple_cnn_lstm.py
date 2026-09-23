"""
ONE simple, self-contained CNN-LSTM model. Deliberately NOT doing the
full multi-variant, multi-hour cross-validated comparison -- this is a
single, correctly-computed evaluation, prioritizing a clean, trustworthy
result over exhaustiveness.

What's included: pooled training (scalable, no per-household retraining
needed for new customers), early stopping, a light 2-config hyperparameter
check, fine-tuning per household, and the corrected MAPE calculation
(each household's own scaler, near-zero values properly excluded).

What's deliberately left out, stated plainly: full cross-validation
across multiple time folds. This gives you ONE honest number, not five
averaged ones -- given every earlier cross-validated result in this
project came out roughly 2x higher than its single-split equivalent,
treat this number as informative but likely somewhat optimistic, same
pattern as every other single-split result so far.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import create_engine
from tensorflow import keras
from tensorflow.keras import layers

from features import (
    load_features, scale_columns, create_sequences, set_seeds, compute_metrics,
    build_next_row, FEATURE_COLUMNS, HORIZON, discover_real_households,
)

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)

LOOKBACK = 24
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "simple_cnn_lstm.keras")
METADATA_PATH = os.path.join(MODEL_DIR, "simple_cnn_lstm_metadata.json")
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 0.0001


def mape(y_true, y_pred, min_actual_threshold: float = 0.05):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = np.abs(y_true) > min_actual_threshold
    excluded = len(y_true) - mask.sum()
    if excluded > 0:
        print(f"    (MAPE: excluded {excluded}/{len(y_true)} near-zero points)")
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def build_cnn_lstm(n_features: int, units: int = 32, filters: int = 32, dropout: float = 0.2):
    model = keras.Sequential([
        layers.Input(shape=(LOOKBACK, n_features)),
        layers.Conv1D(filters=filters, kernel_size=3, activation="relu", padding="causal"),
        layers.LSTM(units),
        layers.Dropout(dropout),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_pooled_train_and_val(engine, household_ids: list):
    X_train_pool, y_train_pool = [], []
    val_per_household = []
    for hid in household_ids:
        df = load_features(engine, hid)
        train_df = df.iloc[:-HORIZON]
        scaled, scalers = scale_columns(train_df)
        X, y = create_sequences(scaled, LOOKBACK)
        split_idx = int(len(X) * 0.9)
        X_train_pool.append(X[:split_idx])
        y_train_pool.append(y[:split_idx])
        val_per_household.append((X[split_idx:], y[split_idx:], scalers["kwh"]))
    return np.concatenate(X_train_pool), np.concatenate(y_train_pool), val_per_household


def evaluate_pooled_mape(model, val_per_household) -> float:
    all_actual, all_pred = [], []
    for X_val, y_val, scaler in val_per_household:
        if len(X_val) == 0:
            continue
        pred_scaled = model.predict(X_val, verbose=0).flatten()
        all_pred.append(scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten())
        all_actual.append(scaler.inverse_transform(y_val.reshape(-1, 1)).flatten())
    return mape(np.concatenate(all_actual), np.concatenate(all_pred))


def train_and_evaluate(engine, household_ids: list, units: int):
    set_seeds()
    X_train, y_train, val_per_household = build_pooled_train_and_val(engine, household_ids)
    X_val = np.concatenate([v[0] for v in val_per_household if len(v[0]) > 0])
    y_val = np.concatenate([v[1] for v in val_per_household if len(v[0]) > 0])

    model = build_cnn_lstm(len(FEATURE_COLUMNS), units=units)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    print(f"Training (units={units}), {len(X_train)} train sequences, {len(X_val)} validation...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=50, batch_size=32, callbacks=[early_stop], verbose=2)

    val_mape = evaluate_pooled_mape(model, val_per_household)
    print(f"  -> pooled validation MAPE: {val_mape:.2f}%")
    return model, val_mape


def fine_tune_and_forecast(model, engine, household_id: int) -> dict:
    set_seeds()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss="mse")

    df = load_features(engine, household_id)
    train_df = df.iloc[:-HORIZON]
    test_df = df.iloc[-HORIZON:]
    scaled_train, scalers = scale_columns(train_df)
    X, y = create_sequences(scaled_train, LOOKBACK)

    split_idx = int(len(X) * 0.9)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(X[:split_idx], y[:split_idx], validation_data=(X[split_idx:], y[split_idx:]),
              epochs=FINE_TUNE_EPOCHS, batch_size=32, callbacks=[early_stop], verbose=0)

    kwh_scaler = scalers["kwh"]
    window = scaled_train[-LOOKBACK:, :].copy()
    last_rolling_and_peak = {
        col: scaled_train[-1, FEATURE_COLUMNS.index(col)]
        for col in ["rolling_7d_mean", "rolling_7d_std", "rolling_30d_mean", "rolling_30d_std", "is_peak"]
    }
    scaled_predictions = []
    for step in range(HORIZON):
        ts = test_df.index[step]
        x_input = window.reshape((1, LOOKBACK, len(FEATURE_COLUMNS)))
        next_kwh_scaled = model.predict(x_input, verbose=0)[0, 0]
        scaled_predictions.append(next_kwh_scaled)
        next_row = build_next_row(next_kwh_scaled, ts, train_df, test_df, step, scalers, last_rolling_and_peak)
        window = np.vstack([window[1:], next_row])

    predicted_kwh = kwh_scaler.inverse_transform(np.array(scaled_predictions).reshape(-1, 1)).flatten()
    metrics = compute_metrics(test_df["kwh"].values, predicted_kwh)
    corrected_mape = mape(test_df["kwh"].values, predicted_kwh)
    metrics["mape_corrected"] = round(corrected_mape, 2)

    predictions = [
        {"timestamp": ts.isoformat(), "predicted_kwh": round(float(p), 4), "actual_kwh": round(float(a), 4)}
        for ts, p, a in zip(test_df.index, predicted_kwh, test_df["kwh"].values)
    ]

    return {"household_id": household_id, "metrics": metrics, "predictions": predictions}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--household-id", type=int, required=True)
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)
    household_ids = discover_real_households(engine)
    print(f"Pooling {len(household_ids)} households: {household_ids}\n")

    print("=== Light hyperparameter check: units=32 vs units=64 ===")
    configs = []
    for units in [32, 64]:
        model, val_mape = train_and_evaluate(engine, household_ids, units)
        configs.append({"units": units, "model": model, "val_mape": val_mape})

    best = min(configs, key=lambda c: c["val_mape"])
    print(f"\nBest config: units={best['units']} (val MAPE={best['val_mape']:.2f}%)")

    os.makedirs(MODEL_DIR, exist_ok=True)
    best["model"].save(MODEL_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump({"units": best["units"], "trained_on_households": household_ids,
                    "trained_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)

    print(f"\n=== Fine-tuning and forecasting for household {args.household_id} ===")
    result = fine_tune_and_forecast(best["model"], engine, args.household_id)
    print(f"Final metrics: {result['metrics']}")

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"simple_cnn_lstm_{args.household_id}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {path}")
