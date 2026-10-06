"""
Rolling-origin cross-validation for the 1-MONTH direct multi-step model.
Until this runs, the 1-month model's real accuracy is genuinely unknown.

Test window here is 720 hours (30 days) per fold -- much larger than the
24-hour window used elsewhere. n_folds may be reduced automatically if a
household's history isn't long enough to fit 5 full 720-hour folds; this
is checked and reported clearly, not silently truncated.
"""
import os

import numpy as np
from sqlalchemy import create_engine
from tensorflow import keras

from features import load_features, scale_columns, set_seeds, compute_metrics, FEATURE_COLUMNS, discover_real_households
from direct_multistep_global import (
    LOOKBACK, HORIZONS, create_direct_multistep_sequences, build_direct_multistep_model,
)
from simple_cnn_lstm import mape

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)

HORIZON = HORIZONS["1month"]


def generate_folds(df, horizon_hours: int, n_folds: int, min_train_hours: int):
    total_len = len(df)
    usable_len = total_len - min_train_hours
    max_folds_that_fit = usable_len // horizon_hours
    if max_folds_that_fit < n_folds:
        print(f"WARNING: requested {n_folds} folds, but only {max_folds_that_fit} fit given "
              f"{total_len} total hours and a {horizon_hours}h test window. Using {max_folds_that_fit} instead.")
        n_folds = max(1, max_folds_that_fit)
    step = usable_len // n_folds
    folds = []
    for i in range(n_folds):
        test_end = min_train_hours + (i + 1) * step
        test_start = test_end - horizon_hours
        folds.append((df.iloc[:test_start], df.iloc[test_start:test_end]))
    return folds, n_folds


def _run_one_fold(engine, other_ids, target_train_df, target_test_df, units):
    keras.backend.clear_session()
    set_seeds()

    X_pool, y_pool = [], []
    for hid in other_ids:
        df = load_features(engine, hid)
        scaled, _ = scale_columns(df)
        X, y = create_direct_multistep_sequences(scaled, LOOKBACK, HORIZON)
        if len(X) == 0:
            continue
        X_pool.append(X)
        y_pool.append(y)

    target_scaled_train, target_scalers = scale_columns(target_train_df)
    X_target, y_target = create_direct_multistep_sequences(target_scaled_train, LOOKBACK, HORIZON)
    if len(X_target) > 0:
        X_pool.append(X_target)
        y_pool.append(y_target)

    X_pool = np.concatenate(X_pool, axis=0)
    y_pool = np.concatenate(y_pool, axis=0)

    split_idx = int(len(X_pool) * 0.9)
    model = build_direct_multistep_model(len(FEATURE_COLUMNS), HORIZON, units=units)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    print(f"  Training pooled base ({len(X_pool)} sequences)...")
    model.fit(X_pool[:split_idx], y_pool[:split_idx],
              validation_data=(X_pool[split_idx:], y_pool[split_idx:]),
              epochs=50, batch_size=32, callbacks=[early_stop], verbose=2)

    if len(X_target) > 0:
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="mse")
        ft_split = max(1, int(len(X_target) * 0.9))
        print(f"  Fine-tuning on target household ({len(X_target)} sequences)...")
        model.fit(X_target[:ft_split], y_target[:ft_split],
                  validation_data=(X_target[ft_split:], y_target[ft_split:]) if ft_split < len(X_target) else None,
                  epochs=10, batch_size=8,
                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)] if ft_split < len(X_target) else [],
                  verbose=0)
    else:
        print("  WARNING: not enough target household data to fine-tune this fold -- using pooled base only.")

    kwh_scaler = target_scalers["kwh"]
    last_window = target_scaled_train[-LOOKBACK:, :].reshape(1, LOOKBACK, len(FEATURE_COLUMNS))
    scaled_forecast = model.predict(last_window, verbose=0).flatten()
    predicted_kwh = kwh_scaler.inverse_transform(scaled_forecast.reshape(-1, 1)).flatten()

    actual_kwh = target_test_df["kwh"].values[:HORIZON]
    metrics = compute_metrics(actual_kwh, predicted_kwh)
    metrics["mape_corrected"] = round(mape(actual_kwh, predicted_kwh), 2)
    return metrics


def cross_validate_1month(engine, target_id: int, units: int = 32, n_folds: int = 5,
                           min_train_hours: int = 5000, pool_limit: int = None) -> dict:
    all_ids = discover_real_households(engine, limit=pool_limit)
    other_ids = [hid for hid in all_ids if hid != target_id]

    target_df = load_features(engine, target_id)
    folds, actual_n_folds = generate_folds(target_df, HORIZON, n_folds, min_train_hours)

    fold_results = []
    for i, (train_df, test_df) in enumerate(folds, 1):
        print(f"\nFold {i}/{actual_n_folds}: target train={len(train_df)}h, test starts {test_df.index[0]}...")
        metrics = _run_one_fold(engine, other_ids, train_df, test_df, units)
        fold_results.append({"fold": i, "train_hours": len(train_df),
                              "test_start": test_df.index[0].isoformat(), "metrics": metrics})
        print(f"  -> MAE={metrics['mae']}, RMSE={metrics['rmse']}, MAPE={metrics['mape']}")

    avg_metrics = {k: round(float(np.mean([f["metrics"][k] for f in fold_results])), 4) for k in ["mae", "rmse", "mape"]}
    std_metrics = {f"{k}_std": round(float(np.std([f["metrics"][k] for f in fold_results])), 4) for k in ["mae", "rmse", "mape"]}
    return {"household_id": target_id, "units": units, "horizon_hours": HORIZON, "n_folds": actual_n_folds,
            "fold_results": fold_results, "avg_metrics": avg_metrics, "std_metrics": std_metrics}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--household-id", type=int, required=True)
    parser.add_argument("--units", type=int, default=32)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--pool-limit", type=int, default=None,
                         help="Limit OTHER households pooled per fold -- use if memory is a constraint")
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)
    print(f"Cross-validating 1-MONTH direct multi-step model for household {args.household_id}")
    print("NOTE: each fold retrains the pooled base model with a 720-hour output -- expect a long run.\n")

    result = cross_validate_1month(engine, args.household_id, units=args.units, n_folds=args.n_folds,
                                    pool_limit=args.pool_limit)

    print(f"\n=== Cross-validated average (across {result['n_folds']} folds) ===")
    print(f"MAE:  {result['avg_metrics']['mae']} (+/- {result['std_metrics']['mae_std']})")
    print(f"RMSE: {result['avg_metrics']['rmse']} (+/- {result['std_metrics']['rmse_std']})")
    print(f"MAPE: {result['avg_metrics']['mape']} (+/- {result['std_metrics']['mape_std']})")

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"direct_multistep_1month_cv_{args.household_id}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {path}")
