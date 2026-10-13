"""
Third CNN-LSTM variant (Bidirectional LSTM) - 1-month, NO-weather half.
Mirrors the other two variants' 1-month companions: single forward pass
over the lookback window produces all 720 future hours at once, no
recursion, no future weather ever read.

Same Conv1D(causal) + Bidirectional(LSTM) body as bidirectional_cnn_lstm_multihorizon.py,
with a Dense(horizon) output head instead of Dense(1). LOOKBACK=24, same
family as the sequential model's own direct_multistep_global.py, not the
day-blocked 168h window the TimeDistributed variant uses.

Tuning fixes applied from the start (not rediscovered): dropout after the
conv block, Huber loss, EarlyStopping(patience=6)+ReduceLROnPlateau, and
scaler persistence via scaler_io.py.

Usage (GPU-first, CPU-fallback-only-if-it-fails, per your standing instruction):
    docker compose exec -T backend python ml/forecasting/cnn_bidirectional/bidirectional_direct_multistep_1month.py --pool-limit 6 --household-id 8162493
    docker compose exec -T -e CUDA_VISIBLE_DEVICES="" backend python ml/forecasting/cnn_bidirectional/bidirectional_direct_multistep_1month.py --pool-limit 6 --household-id 8162493
"""
import argparse
import os

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sqlalchemy import create_engine

# Sibling-folder import shim - see bidirectional_cnn_lstm_multihorizon.py.
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
    discover_real_households,
    set_seeds,
)
from simple_cnn_lstm import mape
from scaler_io import save_scalers, load_scalers, scaler_path_for_household

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)


def apply_scalers(df: pd.DataFrame, scalers: dict) -> np.ndarray:
    scaled = np.zeros((len(df), len(FEATURE_COLUMNS)), dtype=np.float32)
    for i, col in enumerate(FEATURE_COLUMNS):
        scaled[:, i] = scalers[col].transform(df[[col]].values).flatten().astype(np.float32)
    return scaled


LOOKBACK = 24           # same family as the sequential model - flat hourly window
HORIZON = 720           # 30 days x 24 hours
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 0.0001

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "bidirectional_direct_multistep_1month.keras")


def create_direct_multistep_sequences(data: np.ndarray, lookback: int, horizon: int):
    X, y = [], []
    kwh_col_idx = FEATURE_COLUMNS.index("kwh")
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + horizon, kwh_col_idx])
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")


def build_bidirectional_direct_multistep(n_features: int, horizon: int, filters: int = 32, units: int = 32, dropout: float = 0.2):
    """Same Conv1D(causal)+Bidirectional(LSTM) body as the recursive model,
    Dense(horizon) head instead of Dense(1) - one shot, no recursion, no
    future weather ever read."""
    inputs = layers.Input(shape=(LOOKBACK, n_features))
    x = layers.Conv1D(filters, kernel_size=3, padding="causal", activation="relu")(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(units, dropout=dropout))(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(horizon)(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="huber")
    return model


def build_pooled_train_and_val(engine, household_ids):
    X_train_all, y_train_all = [], []
    X_val_all, y_val_all = [], []

    for hh_id in household_ids:
        df = load_features(engine, hh_id)
        if len(df) < LOOKBACK + HORIZON + 48:
            continue

        split_idx = int(len(df) * 0.9)
        train_df, val_df = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
        if len(val_df) < LOOKBACK + HORIZON:
            val_df = None

        train_scaled, scaler = scale_columns(train_df)
        X_tr, y_tr = create_direct_multistep_sequences(train_scaled, LOOKBACK, HORIZON)
        if len(X_tr) == 0:
            continue
        X_train_all.append(X_tr)
        y_train_all.append(y_tr)

        if val_df is not None:
            val_scaled = apply_scalers(val_df, scaler)
            X_va, y_va = create_direct_multistep_sequences(val_scaled, LOOKBACK, HORIZON)
            if len(X_va) > 0:
                X_val_all.append(X_va)
                y_val_all.append(y_va)

    X_train = np.concatenate(X_train_all).astype("float32")
    y_train = np.concatenate(y_train_all).astype("float32")
    X_val = np.concatenate(X_val_all).astype("float32") if X_val_all else X_train[-200:]
    y_val = np.concatenate(y_val_all).astype("float32") if y_val_all else y_train[-200:]
    return X_train, y_train, X_val, y_val


def fine_tune_and_forecast(model, engine, household_id: int):
    df = load_features(engine, household_id)
    if len(df) < LOOKBACK + HORIZON + 48:
        raise ValueError(f"Not enough history for household {household_id} (need >= {LOOKBACK + HORIZON + 48} hourly rows)")

    train_df = df.iloc[:-HORIZON].copy()
    test_df = df.iloc[-HORIZON:].copy()

    scaled_train, scaler = scale_columns(train_df)
    X_ft, y_ft = create_direct_multistep_sequences(scaled_train, LOOKBACK, HORIZON)
    if len(X_ft) == 0:
        raise ValueError(f"Not enough history to fine-tune for household {household_id}")

    save_scalers(scaler, scaler_path_for_household(MODEL_DIR, "bidirectional_1month", household_id))

    ft_model = keras.models.clone_model(model)
    ft_model.set_weights(model.get_weights())
    ft_model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss="huber")
    ft_model.fit(X_ft, y_ft, epochs=FINE_TUNE_EPOCHS, batch_size=16, verbose=0)

    last_window_unscaled = train_df.iloc[-LOOKBACK:]
    last_window_scaled = apply_scalers(last_window_unscaled, scaler)
    x = last_window_scaled.reshape(1, LOOKBACK, len(FEATURE_COLUMNS))

    pred_scaled = ft_model.predict(x, verbose=0)[0]
    pred_kwh = scaler["kwh"].inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    future_index = test_df.index[:HORIZON]
    predictions = [
        {"timestamp": ts.isoformat(), "predicted_kwh": round(float(v), 4), "actual_kwh": round(float(a), 4)}
        for ts, v, a in zip(future_index, pred_kwh, test_df["kwh"].values[:HORIZON])
    ]

    y_true = test_df["kwh"].values[:HORIZON]
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    metrics = {
        "mae": round(float(mean_absolute_error(y_true, pred_kwh)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, pred_kwh))), 4),
        "mape": round(float(mape(y_true, pred_kwh)), 2),
    }
    return {"household_id": household_id, "horizon_hours": HORIZON, "metrics": metrics, "predictions": predictions}


def load_saved_scaler_and_model(household_id: int):
    model = keras.models.load_model(MODEL_PATH)
    scaler = load_scalers(scaler_path_for_household(MODEL_DIR, "bidirectional_1month", household_id))
    return model, scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-limit", type=int, default=6,
                         help="Cap households pooled for base training")
    parser.add_argument("--household-id", type=int, default=None)
    args = parser.parse_args()

    set_seeds(42)
    os.makedirs(MODEL_DIR, exist_ok=True)

    engine = create_engine(DATABASE_URL)
    household_ids = discover_real_households(engine, limit=args.pool_limit)
    print(f"Pooling {len(household_ids)} households for 1-month direct multistep (pool-limit={args.pool_limit})")

    keras.backend.clear_session()

    X_train, y_train, X_val, y_val = build_pooled_train_and_val(engine, household_ids)
    print(f"Train sequences: {X_train.shape}, Val sequences: {X_val.shape}")

    model = build_bidirectional_direct_multistep(n_features=len(FEATURE_COLUMNS), horizon=HORIZON)
    model.summary()

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20, batch_size=16,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5, verbose=1),
        ],
        verbose=1,
    )

    val_pred = model.predict(X_val, verbose=0)
    pooled_mape = mape(y_val.flatten(), val_pred.flatten())
    print(f"Pooled validation MAPE (flattened across all 720 steps): {pooled_mape:.2f}%")

    model.save(MODEL_PATH)
    print(f"Saved pooled base model to {MODEL_PATH}")

    target_household = args.household_id or household_ids[0]
    print(f"\n=== Fine-tune + 1-month forecast: household {target_household} ===")
    result = fine_tune_and_forecast(model, engine, target_household)
    print(f"Metrics: {result['metrics']}")
    print(f"First 3 predictions: {result['predictions'][:3]}")
    print(f"Last 3 predictions:  {result['predictions'][-3:]}")


if __name__ == "__main__":
    main()
