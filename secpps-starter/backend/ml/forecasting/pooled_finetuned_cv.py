"""
Rolling-origin cross-validation for the POOLED+FINE-TUNED model --
closes the same proposal gap as lstm_cv.py, but for your actual
production model (pooled), not the per-household one.

Per fold: pool = [all OTHER households' full data] + [target household's
own train portion up to this fold's cutoff]. Fine-tune on target's
fold-train portion. Evaluate on target's fold-test window. Other
households' full data is safe to include in full since they are not
being evaluated in this fold -- no leakage against the metric computed.
"""
import os

import numpy as np
from sqlalchemy import create_engine
from tensorflow import keras

from lstm_forecast import set_seeds, compute_metrics
from lstm_full_features import (
    load_features_with_weather, scale_columns, create_sequences, build_model,
    FEATURE_COLUMNS, LOOKBACK,
)
from pooled_finetuned_full_features_lstm import discover_real_households, _build_next_row, FINE_TUNE_EPOCHS, FINE_TUNE_LR

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


def _run_one_fold(engine, target_id, other_ids, target_train_df, target_test_df, lookback, horizon):
    set_seeds()

    X_pool, y_pool = [], []
    for hid in other_ids:
        df = load_features_with_weather(engine, hid)
        scaled, _ = scale_columns(df)
        X, y = create_sequences(scaled, lookback)
        X_pool.append(X)
        y_pool.append(y)

    target_scaled_train, target_scalers = scale_columns(target_train_df)
    X_target, y_target = create_sequences(target_scaled_train, lookback)
    X_pool.append(X_target)
    y_pool.append(y_target)

    X_pool = np.concatenate(X_pool, axis=0)
    y_pool = np.concatenate(y_pool, axis=0)

    split_idx = int(len(X_pool) * 0.9)
    model = build_model(n_features=len(FEATURE_COLUMNS), lookback=lookback)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_pool[:split_idx], y_pool[:split_idx],
              validation_data=(X_pool[split_idx:], y_pool[split_idx:]),
              epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss="mse")
    ft_split = int(len(X_target) * 0.9)
    ft_early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(X_target[:ft_split], y_target[:ft_split],
              validation_data=(X_target[ft_split:], y_target[ft_split:]),
              epochs=FINE_TUNE_EPOCHS, batch_size=32, callbacks=[ft_early_stop], verbose=0)

    kwh_scaler = target_scalers["kwh"]
    window = target_scaled_train[-lookback:, :].copy()
    last_rolling_and_peak = {
        col: target_scaled_train[-1, FEATURE_COLUMNS.index(col)]
        for col in ["rolling_7d_mean", "rolling_7d_std", "rolling_30d_mean", "rolling_30d_std", "is_peak"]
    }

    scaled_predictions = []
    for step in range(horizon):
        ts = target_test_df.index[step]
        x_input = window.reshape((1, lookback, len(FEATURE_COLUMNS)))
        next_kwh_scaled = model.predict(x_input, verbose=0)[0, 0]
        scaled_predictions.append(next_kwh_scaled)
        next_row = _build_next_row(next_kwh_scaled, ts, target_train_df, target_test_df, step,
                                    target_scalers, last_rolling_and_peak)
        window = np.vstack([window[1:], next_row])

    predicted_kwh = kwh_scaler.inverse_transform(np.array(scaled_predictions).reshape(-1, 1)).flatten()
    return compute_metrics(target_test_df["kwh"].values[:horizon], predicted_kwh)


def cross_validate_pooled(engine, target_id: int, horizon_hours: int = 24, n_folds: int = 5,
                           min_train_hours: int = 2000, lookback: int = LOOKBACK) -> dict:
    all_ids = discover_real_households(engine)
    other_ids = [hid for hid in all_ids if hid != target_id]

    target_df = load_features_with_weather(engine, target_id)
    folds = generate_folds(target_df, horizon_hours, n_folds, min_train_hours)

    fold_results = []
    for i, (train_df, test_df) in enumerate(folds, 1):
        print(f"Fold {i}/{n_folds}: target train={len(train_df)}h, test starts {test_df.index[0]}...")
        metrics = _run_one_fold(engine, target_id, other_ids, train_df, test_df, lookback, horizon_hours)
        fold_results.append({"fold": i, "train_hours": len(train_df),
                              "test_start": test_df.index[0].isoformat(), "metrics": metrics})
        print(f"  -> MAE={metrics['mae']}, RMSE={metrics['rmse']}, MAPE={metrics['mape']}")

    avg_metrics = {
        "mae": round(float(np.mean([f["metrics"]["mae"] for f in fold_results])), 4),
        "rmse": round(float(np.mean([f["metrics"]["rmse"] for f in fold_results])), 4),
        "mape": round(float(np.mean([f["metrics"]["mape"] for f in fold_results])), 2),
    }
    std_metrics = {
        "mae_std": round(float(np.std([f["metrics"]["mae"] for f in fold_results])), 4),
        "rmse_std": round(float(np.std([f["metrics"]["rmse"] for f in fold_results])), 4),
        "mape_std": round(float(np.std([f["metrics"]["mape"] for f in fold_results])), 2),
    }
    return {"household_id": target_id, "n_folds": n_folds, "fold_results": fold_results,
            "avg_metrics": avg_metrics, "std_metrics": std_metrics}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--household-id", type=int, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)
    print(f"Cross-validating POOLED+FINE-TUNED LSTM for household {args.household_id}, {args.n_folds} folds:\n")
    print("NOTE: each fold retrains the full pooled base model -- this will take considerably "
          "longer than the per-household CV script.\n")

    result = cross_validate_pooled(engine, args.household_id, n_folds=args.n_folds)

    print(f"\n=== Cross-validated average (across {result['n_folds']} folds) ===")
    print(f"MAE:  {result['avg_metrics']['mae']} (+/- {result['std_metrics']['mae_std']})")
    print(f"RMSE: {result['avg_metrics']['rmse']} (+/- {result['std_metrics']['rmse_std']})")
    print(f"MAPE: {result['avg_metrics']['mape']} (+/- {result['std_metrics']['mape_std']})")

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"pooled_finetuned_cv_{args.household_id}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {path}")
