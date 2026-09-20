"""
Direct multi-step version of the pooled+fine-tuned model: Dense(24)
output, one forward pass, no recursive error compounding.

SAME TRADE-OFF as lstm_direct_multistep.py: no mechanism to inject
future weather/generation values -- relies solely on the historical
window. Report whichever architecture the evidence favours.
"""
import json
import os

import numpy as np
from sqlalchemy import create_engine
from tensorflow import keras
from tensorflow.keras import layers

from lstm_forecast import set_seeds, compute_metrics
from lstm_full_features import load_features_with_weather, scale_columns, FEATURE_COLUMNS, LOOKBACK, HORIZON
from pooled_finetuned_full_features_lstm import discover_real_households, FINE_TUNE_EPOCHS, FINE_TUNE_LR

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pooled_multistep_lstm.keras")
METADATA_PATH = os.path.join(MODEL_DIR, "pooled_multistep_metadata.json")


def create_multistep_sequences(data: np.ndarray, lookback: int, horizon: int):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i + lookback, :])
        y.append(data[i + lookback: i + lookback + horizon, 0])
    return np.array(X), np.array(y)


def build_multistep_model(n_features: int, lookback: int, horizon: int, units: int = 32, dropout: float = 0.2):
    model = keras.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.LSTM(units),
        layers.Dropout(dropout),
        layers.Dense(horizon),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_pooled_multistep_base(engine, household_ids: list, lookback: int = LOOKBACK, horizon: int = HORIZON):
    set_seeds()

    X_pool, y_pool = [], []
    for hid in household_ids:
        df = load_features_with_weather(engine, hid)
        train_df = df.iloc[:-horizon]
        scaled, _ = scale_columns(train_df)
        X, y = create_multistep_sequences(scaled, lookback, horizon)
        X_pool.append(X)
        y_pool.append(y)

    X_pool = np.concatenate(X_pool, axis=0)
    y_pool = np.concatenate(y_pool, axis=0)
    print(f"Total pooled multistep sequences: {len(X_pool)}")

    split_idx = int(len(X_pool) * 0.9)
    model = build_multistep_model(len(FEATURE_COLUMNS), lookback, horizon)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(X_pool[:split_idx], y_pool[:split_idx],
                         validation_data=(X_pool[split_idx:], y_pool[split_idx:]),
                         epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)
    print(f"Base training complete after {len(history.history['loss'])} epochs, "
          f"final val loss: {history.history['val_loss'][-1]:.6f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump({"trained_on_households": household_ids, "lookback": lookback, "horizon": horizon}, f, indent=2)
    print(f"Saved to {MODEL_PATH}")


def fine_tune_and_forecast_multistep(engine, household_id: int) -> dict:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Run --mode train first")

    set_seeds()
    model = keras.models.load_model(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    lookback, horizon = metadata["lookback"], metadata["horizon"]

    df = load_features_with_weather(engine, household_id)
    train_df = df.iloc[:-horizon]
    test_df = df.iloc[-horizon:]

    scaled_train, scalers = scale_columns(train_df)
    X, y = create_multistep_sequences(scaled_train, lookback, horizon)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss="mse")
    split_idx = int(len(X) * 0.9)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(X[:split_idx], y[:split_idx], validation_data=(X[split_idx:], y[split_idx:]),
              epochs=FINE_TUNE_EPOCHS, batch_size=32, callbacks=[early_stop], verbose=0)

    last_window = scaled_train[-lookback:, :].reshape((1, lookback, len(FEATURE_COLUMNS)))
    scaled_predictions = model.predict(last_window, verbose=0)[0]

    kwh_scaler = scalers["kwh"]
    predicted_kwh = kwh_scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).flatten()
    metrics = compute_metrics(test_df["kwh"].values, predicted_kwh)

    predictions = [
        {"timestamp": ts.isoformat(), "predicted_kwh": round(float(v), 4)}
        for ts, v in zip(test_df.index, predicted_kwh)
    ]

    return {
        "household_id": household_id,
        "model": "Pooled + fine-tuned LSTM, direct multi-step (Dense(24))",
        "was_in_training_pool": household_id in metadata["trained_on_households"],
        "predictions": predictions, "metrics": metrics,
        "trade_off": "No future weather/generation input -- relies solely on historical window.",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "forecast"], required=True)
    parser.add_argument("--household-ids", type=str)
    parser.add_argument("--household-id", type=int)
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)

    if args.mode == "train":
        household_ids = [int(x) for x in args.household_ids.split(",")] if args.household_ids else discover_real_households(engine)
        print(f"Training on: {household_ids}")
        train_pooled_multistep_base(engine, household_ids)
    else:
        result = fine_tune_and_forecast_multistep(engine, args.household_id)
        print(f"\nHousehold {result['household_id']}")
        print(f"Metrics: {result['metrics']}")
        print(f"Trade-off: {result['trade_off']}")

        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, f"pooled_multistep_{args.household_id}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {path}")
