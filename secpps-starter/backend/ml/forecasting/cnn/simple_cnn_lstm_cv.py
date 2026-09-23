"""
Rolling-origin cross-validation for simple_cnn_lstm.py's winning
configuration (sequential Conv1D-LSTM, units=32). Same fold structure as
every other CV script in this project: each fold retrains the full
pooled base model using all OTHER households' complete history plus the
target household's own data up to that fold's cutoff, then fine-tunes
and evaluates on the target's fold-test window.

Given every single-split result in this project has come out roughly
half its true cross-validated value, treat the earlier single-split
result (MAE 0.1931) as optimistic until this confirms or corrects it.
"""
import os

import numpy as np
from sqlalchemy import create_engine
from tensorflow import keras

from features import (
    load_features, scale_columns, create_sequences, set_seeds, compute_metrics,
    build_next_row, FEATURE_COLUMNS, discover_real_households,
)
from simple_cnn_lstm import build_cnn_lstm, mape, LOOKBACK, FINE_TUNE_EPOCHS, FINE_TUNE_LR

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)


def generate_folds(df, horizon_hours: int, n_folds: int, min_train_hours: int):
    total_len = len(df)
    usable_len = total_len - min_train_hours
    if usable_len < horizon_hours * n_folds:
        raise ValueError(f"Not enough data for {n_folds} folds. Length: {total_len}")
    step = usable_len // n_folds
    folds = []
    for i in range(n_folds):
        test_end = min_train_hours + (i + 1) * step
        test_start = test_end - horizon_hours
        folds.append((df.iloc[:test_start], df.iloc[test_start:test_end]))
    return folds


def _run_one_fold(engine, other_ids, target_train_df, target_test_df, units, horizon):
    set_seeds()

    X_pool, y_pool = [], []
    for hid in other_ids:
        df = load_features(engine, hid)
        scaled, _ = scale_columns(df)
        X, y = create_sequences(scaled, LOOKBACK)
        X_pool.append(X)
        y_pool.append(y)

    target_scaled_train, target_scalers = scale_columns(target_train_df)
    X_target, y_target = create_sequences(target_scaled_train, LOOKBACK)
    X_pool.append(X_target)
    y_pool.append(y_target)

    X_pool = np.concatenate(X_pool, axis=0)
    y_pool = np.concatenate(y_pool, axis=0)

    split_idx = int(len(X_pool) * 0.9)
    model = build_cnn_lstm(len(FEATURE_COLUMNS), units=units)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    print(f"  Training pooled base ({len(X_pool)} sequences)...")
    model.fit(X_pool[:split_idx], y_pool[:split_idx],
              validation_data=(X_pool[split_idx:], y_pool[split_idx:]),
              epochs=50, batch_size=32, callbacks=[early_stop], verbose=2)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss="mse")
    ft_split = int(len(X_target) * 0.9)
    ft_early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    print(f"  Fine-tuning on target household ({len(X_target)} sequences)...")
    model.fit(X_target[:ft_split], y_target[:ft_split],
              validation_data=(X_target[ft_split:], y_target[ft_split:]),
              epochs=FINE_TUNE_EPOCHS, batch_size=32, callbacks=[ft_early_stop], verbose=0)

    kwh_scaler = target_scalers["kwh"]
    window = target_scaled_train[-LOOKBACK:, :].copy()
    last_rolling_and_peak = {
        col: target_scaled_train[-1, FEATURE_COLUMNS.index(col)]
        for col in ["rolling_7d_mean", "rolling_7d_std", "rolling_30d_mean", "rolling_30d_std", "is_peak"]
    }

    scaled_predictions = []
    for step in range(horizon):
        ts = target_test_df.index[step]
        x_input = window.reshape((1, LOOKBACK, len(FEATURE_COLUMNS)))
        next_kwh_scaled = model.predict(x_input, verbose=0)[0, 0]
        scaled_predictions.append(next_kwh_scaled)
        next_row = build_next_row(next_kwh_scaled, ts, target_train_df, target_test_df, step,
                                   target_scalers, last_rolling_and_peak)
        window = np.vstack([window[1:], next_row])

    predicted_kwh = kwh_scaler.inverse_transform(np.array(scaled_predictions).reshape(-1, 1)).flatten()
    metrics = compute_metrics(target_test_df["kwh"].values[:horizon], predicted_kwh)
    metrics["mape_corrected"] = round(mape(target_test_df["kwh"].values[:horizon], predicted_kwh), 2)
    return metrics


def cross_validate_simple_cnn_lstm(engine, target_id: int, units: int = 32,
                                    horizon_hours: int = 24, n_folds: int = 5,
                                    min_train_hours: int = 2000) -> dict:
    all_ids = discover_real_households(engine)
    other_ids = [hid for hid in all_ids if hid != target_id]

    target_df = load_features(engine, target_id)
    folds = generate_folds(target_df, horizon_hours, n_folds, min_train_hours)

    fold_results = []
    for i, (train_df, test_df) in enumerate(folds, 1):
        print(f"\nFold {i}/{n_folds}: target train={len(train_df)}h, test starts {test_df.index[0]}...")
        metrics = _run_one_fold(engine, other_ids, train_df, test_df, units, horizon_hours)
        fold_results.append({"fold": i, "train_hours": len(train_df),
                              "test_start": test_df.index[0].isoformat(), "metrics": metrics})
        print(f"  -> MAE={metrics['mae']}, RMSE={metrics['rmse']}, MAPE={metrics['mape']}")

    avg_metrics = {k: round(float(np.mean([f["metrics"][k] for f in fold_results])), 4) for k in ["mae", "rmse", "mape"]}
    std_metrics = {f"{k}_std": round(float(np.std([f["metrics"][k] for f in fold_results])), 4) for k in ["mae", "rmse", "mape"]}
    return {"household_id": target_id, "units": units, "n_folds": n_folds,
            "fold_results": fold_results, "avg_metrics": avg_metrics, "std_metrics": std_metrics}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--household-id", type=int, required=True)
    parser.add_argument("--units", type=int, default=32)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)
    print(f"Cross-validating simple CNN-LSTM (units={args.units}) for household {args.household_id}")
    print("NOTE: each fold retrains the full 30-household pooled base model -- expect several hours total.\n")

    result = cross_validate_simple_cnn_lstm(engine, args.household_id, units=args.units, n_folds=args.n_folds)

    print(f"\n=== Cross-validated average (across {result['n_folds']} folds) ===")
    print(f"MAE:  {result['avg_metrics']['mae']} (+/- {result['std_metrics']['mae_std']})")
    print(f"RMSE: {result['avg_metrics']['rmse']} (+/- {result['std_metrics']['rmse_std']})")
    print(f"MAPE: {result['avg_metrics']['mape']} (+/- {result['std_metrics']['mape_std']})")

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"simple_cnn_lstm_cv_{args.household_id}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {path}")
