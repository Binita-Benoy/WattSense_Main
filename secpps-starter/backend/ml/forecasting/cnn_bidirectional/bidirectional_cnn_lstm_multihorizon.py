
import argparse
import os

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import create_engine

# This folder is a sibling of the other variants' folders under ml/forecasting/,
# so features.py / simple_cnn_lstm.py / scaler_io.py live next door, not here.
# Append (not prepend) any sibling folder holding one of them to sys.path, so
# local copies, if you add them later, still win and nothing else changes.
import sys
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sib in sorted(os.listdir(_PARENT)):
    _p = os.path.join(_PARENT, _sib)
    if os.path.isdir(_p) and _p not in sys.path and any(
        os.path.exists(os.path.join(_p, f)) for f in ("features.py", "simple_cnn_lstm.py", "scaler_io.py")
    ):
        sys.path.append(_p)

from features import (
    FEATURE_COLUMNS,
    load_features,
    scale_columns,
    create_sequences,
    compute_calendar_features,
    build_next_row,
    discover_real_households,
    set_seeds,
)
from simple_cnn_lstm import mape
from scaler_io import save_scalers, load_scalers, scaler_path_for_household

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)


def apply_scalers(df: pd.DataFrame, scalers: dict) -> np.ndarray:
    """Transform-only helper - scale_columns() always fit_transform()s, so
    this is required for applying a train-fitted scaler to val/future rows.
    Same rationale as the identical helper in both earlier variants."""
    scaled = np.zeros((len(df), len(FEATURE_COLUMNS)), dtype=np.float32)
    for i, col in enumerate(FEATURE_COLUMNS):
        scaled[:, i] = scalers[col].transform(df[[col]].values).flatten().astype(np.float32)
    return scaled


LOOKBACK = 24           # flat hourly window - same family as the sequential model, not day-blocked
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 0.0001

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "bidirectional_cnn_lstm_pooled.keras")


def build_bidirectional_cnn_lstm(n_features: int, units: int = 32, filters: int = 32, dropout: float = 0.2):
    """Conv1D (causal padding - no leakage from future steps within the
    window) feeding a Bidirectional(LSTM) instead of the sequential model's
    plain forward-only LSTM. Dropout after the conv block and Huber loss are
    carried straight over from the TimeDistributed variant's tuning pass -
    no reason to re-discover the same fix a third time."""
    inputs = layers.Input(shape=(LOOKBACK, n_features))
    x = layers.Conv1D(filters, kernel_size=3, padding="causal", activation="relu")(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(units, dropout=dropout))(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="huber")
    return model


def build_pooled_train_and_val(engine, household_ids):
    """Same pooling pattern as every other model in this project: per-household
    90/10 split, per-household fitted scaler fit on train only, applied
    transform-only to that household's own val slice."""
    X_train_all, y_train_all = [], []
    X_val_all, y_val_all = [], []
    val_scalers = {}

    for hh_id in household_ids:
        df = load_features(engine, hh_id)
        if len(df) < LOOKBACK + 48:
            continue

        split_idx = int(len(df) * 0.9)
        train_df, val_df = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

        train_scaled, scaler = scale_columns(train_df)
        val_scaled = apply_scalers(val_df, scaler)

        X_tr, y_tr = create_sequences(train_scaled, LOOKBACK)
        X_va, y_va = create_sequences(val_scaled, LOOKBACK)
        if len(X_tr) == 0 or len(X_va) == 0:
            continue

        X_train_all.append(X_tr)
        y_train_all.append(y_tr)
        X_val_all.append(X_va)
        y_val_all.append(y_va)
        val_scalers[hh_id] = scaler

    X_train = np.concatenate(X_train_all).astype("float32")
    y_train = np.concatenate(y_train_all).astype("float32")
    X_val = np.concatenate(X_val_all).astype("float32")
    y_val = np.concatenate(y_val_all).astype("float32")
    return X_train, y_train, X_val, y_val, val_scalers


def fine_tune_and_forecast_recursive(model, engine, household_id: int, horizon_hours: int):
    """Thin wrapper, same split as the TimeDistributed variant: loads full
    history, hands off to the df-based core so the CV script can call the
    core directly on a fold-truncated slice."""
    df = load_features(engine, household_id)
    if len(df) < LOOKBACK + horizon_hours + 48:
        raise ValueError(f"Not enough history for household {household_id} at horizon={horizon_hours}h")
    return _fine_tune_and_forecast_recursive_on_df(model, df, household_id, horizon_hours)


def _fine_tune_and_forecast_recursive_on_df(model, df, household_id: int, horizon_hours: int):
    """Fine-tunes on history, then recursively forecasts horizon_hours
    forward using real held-out weather (offline/CV-style evaluation -
    matches the identical function in the TimeDistributed variant)."""
    test_df = df.iloc[-horizon_hours:].copy()
    history_df = df.iloc[:-horizon_hours].copy()

    scaled_history, scaler = scale_columns(history_df)
    fine_tune_X, fine_tune_y = create_sequences(scaled_history, LOOKBACK)
    if len(fine_tune_X) == 0:
        raise ValueError(f"Not enough history to fine-tune for household {household_id}")

    # Scaler persistence (closes the gap flagged in the sequential model's
    # own report) - saved once per household right after it's fitted here,
    # not wired into the CV script for the same reason as the other two
    # variants: CV must keep refitting a fresh scaler per fold to be valid.
    save_scalers(scaler, scaler_path_for_household(MODEL_DIR, "bidirectional_recursive", household_id))

    ft_model = keras.models.clone_model(model)
    ft_model.set_weights(model.get_weights())
    ft_model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss="huber")
    ft_model.fit(fine_tune_X, fine_tune_y, epochs=FINE_TUNE_EPOCHS, batch_size=32, verbose=0)

    window = scaled_history[-LOOKBACK:].astype("float32")

    ROLLING_PEAK_COLS = ["rolling_7d_mean", "rolling_7d_std", "rolling_30d_mean", "rolling_30d_std", "is_peak"]
    last_row_scaled = scaled_history[-1]
    last_rolling_and_peak = {col: last_row_scaled[FEATURE_COLUMNS.index(col)] for col in ROLLING_PEAK_COLS}

    predictions = []
    for i in range(horizon_hours):
        x = window.reshape(1, LOOKBACK, len(FEATURE_COLUMNS))
        pred_scaled = ft_model.predict(x, verbose=0)[0, 0]

        future_ts = test_df.index[i]
        actual_row = test_df.iloc[i]

        new_row = build_next_row(
            next_kwh_scaled=pred_scaled,
            ts=future_ts,
            train_df=history_df,
            test_df=test_df,
            step=i,
            scalers=scaler,
            last_rolling_and_peak=last_rolling_and_peak,
        ).astype("float32")

        window = np.vstack([window[1:], new_row])

        predicted_kwh = scaler["kwh"].inverse_transform([[pred_scaled]])[0, 0]
        predictions.append({
            "timestamp": future_ts.isoformat(),
            "predicted_kwh": round(float(predicted_kwh), 4),
            "actual_kwh": round(float(actual_row["kwh"]), 4),
        })

    y_pred = np.array([p["predicted_kwh"] for p in predictions])
    y_true = np.array([p["actual_kwh"] for p in predictions])
    metrics = {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mape": round(float(mape(y_true, y_pred)), 2),
    }
    return {"household_id": household_id, "horizon_hours": horizon_hours, "metrics": metrics, "predictions": predictions}


def load_saved_scaler_and_model(household_id: int):
    """Deployment-without-DB helper, same pattern as both earlier variants."""
    model = keras.models.load_model(MODEL_PATH)
    scaler = load_scalers(scaler_path_for_household(MODEL_DIR, "bidirectional_recursive", household_id))
    return model, scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-limit", type=int, default=6,
                         help="Cap households pooled for base training (6 is the value that avoided OOM "
                              "elsewhere in this project - was 10 by default before that was learned)")
    parser.add_argument("--household-id", type=int, default=None,
                         help="Fine-tune + evaluate on this household after pooled training. Defaults to the first discovered.")
    args = parser.parse_args()

    set_seeds(42)
    os.makedirs(MODEL_DIR, exist_ok=True)

    engine = create_engine(DATABASE_URL)
    household_ids = discover_real_households(engine, limit=args.pool_limit)
    print(f"Pooling {len(household_ids)} households (pool-limit={args.pool_limit})")

    keras.backend.clear_session()

    X_train, y_train, X_val, y_val, _ = build_pooled_train_and_val(engine, household_ids)
    print(f"Train sequences: {X_train.shape}, Val sequences: {X_val.shape}")

    model = build_bidirectional_cnn_lstm(n_features=len(FEATURE_COLUMNS))
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30, batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5, verbose=1),
        ],
        verbose=1,
    )

    val_pred = model.predict(X_val, verbose=0).flatten()
    pooled_mape = mape(y_val, val_pred)
    print(f"Pooled validation MAPE: {pooled_mape:.2f}%")

    model.save(MODEL_PATH)
    print(f"Saved pooled base model to {MODEL_PATH}")

    target_household = args.household_id or household_ids[0]
    for horizon_name, horizon_hours in [("24h", 24), ("7day", 168)]:
        print(f"\n=== Fine-tune + forecast: household {target_household}, horizon={horizon_name} ===")
        result = fine_tune_and_forecast_recursive(model, engine, target_household, horizon_hours)
        print(f"Metrics ({horizon_name}): {result['metrics']}")
        print(f"First 3 predictions: {result['predictions'][:3]}")


if __name__ == "__main__":
    main()
