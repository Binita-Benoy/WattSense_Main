"""
5-fold rolling-origin CV for the Bidirectional LSTM variant's 1-month,
NO-weather horizon. Same CV pattern as the other two variants' 1-month CV
scripts: each fold retrains the full pooled base model using all other
households' complete history plus the target household's data up to that
fold's cutoff, then fine-tunes and does ONE single-shot 720-hour prediction
into that fold's held-out window. No recursion, no future weather read
anywhere.

Memory fixes (gc.collect() + max_target_rows cap) applied from the start -
same reasoning as this variant's recursive CV script.

Usage (GPU-first, CPU-fallback-only-if-it-fails, per your standing instruction):
    docker compose exec -T backend python ml/forecasting/cnn_bidirectional/bidirectional_direct_multistep_1month_cv.py --household-id 8162493 --pool-limit 6
    docker compose exec -T -e CUDA_VISIBLE_DEVICES="" backend python ml/forecasting/cnn_bidirectional/bidirectional_direct_multistep_1month_cv.py --household-id 8162493 --pool-limit 6
"""
import argparse
import gc
import os

import numpy as np
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
from bidirectional_direct_multistep_1month import (
    LOOKBACK,
    HORIZON,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LR,
    build_bidirectional_direct_multistep,
    create_direct_multistep_sequences,
    apply_scalers,
)

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)


def generate_folds(total_hours: int, test_window_hours: int, n_folds: int, min_train_hours: int) -> list:
    last_cutoff = total_hours - test_window_hours
    if last_cutoff < min_train_hours:
        raise ValueError(
            f"Not enough history for even one fold: {total_hours} total hours, "
            f"need >= {min_train_hours + test_window_hours}"
        )

    max_possible_folds = max(1, (last_cutoff - min_train_hours) // test_window_hours + 1)
    actual_folds = min(n_folds, max_possible_folds)
    if actual_folds < n_folds:
        print(f"WARNING: only {actual_folds} fold(s) fit this household's history "
              f"(requested {n_folds}) - results will average over fewer folds than asked.")

    if actual_folds == 1:
        return [last_cutoff]

    step = (last_cutoff - min_train_hours) // (actual_folds - 1)
    return [min_train_hours + i * step for i in range(actual_folds)]


def build_pooled_train_and_val_cv(engine, pool_household_ids: list, target_df_up_to_cutoff,
                                   max_target_rows: int = 5000):
    X_train_all, y_train_all = [], []
    X_val_all, y_val_all = [], []

    if len(target_df_up_to_cutoff) > max_target_rows:
        target_df_up_to_cutoff = target_df_up_to_cutoff.iloc[-max_target_rows:]

    for hh_id in pool_household_ids:
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

    if len(target_df_up_to_cutoff) >= LOOKBACK + HORIZON + 48:
        tgt_scaled, _ = scale_columns(target_df_up_to_cutoff)
        X_tgt, y_tgt = create_direct_multistep_sequences(tgt_scaled, LOOKBACK, HORIZON)
        if len(X_tgt) > 0:
            X_train_all.append(X_tgt)
            y_train_all.append(y_tgt)

    X_train = np.concatenate(X_train_all).astype("float32")
    y_train = np.concatenate(y_train_all).astype("float32")
    X_val = np.concatenate(X_val_all).astype("float32") if X_val_all else X_train[-200:]
    y_val = np.concatenate(y_val_all).astype("float32") if y_val_all else y_train[-200:]
    return X_train, y_train, X_val, y_val


def fine_tune_and_forecast_1month_on_df(model, fold_df, household_id: int):
    train_df = fold_df.iloc[:-HORIZON].copy()
    test_df = fold_df.iloc[-HORIZON:].copy()

    scaled_train, scaler = scale_columns(train_df)
    X_ft, y_ft = create_direct_multistep_sequences(scaled_train, LOOKBACK, HORIZON)
    if len(X_ft) == 0:
        raise ValueError(f"Not enough history to fine-tune for household {household_id} on this fold")

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
    metrics = {
        "mae": round(float(mean_absolute_error(y_true, pred_kwh)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, pred_kwh))), 4),
        "mape": round(float(mape(y_true, pred_kwh)), 2),
    }
    return {"household_id": household_id, "horizon_hours": HORIZON, "metrics": metrics, "predictions": predictions}


def summarize(results: list, label: str):
    maes = [r["mae"] for r in results]
    rmses = [r["rmse"] for r in results]
    mapes = [r["mape"] for r in results]
    print(f"\n=== {label} ({len(results)} fold{'s' if len(results) != 1 else ''}) ===")
    print(f"MAE:  {np.mean(maes):.4f} (+/- {np.std(maes):.4f})")
    print(f"RMSE: {np.mean(rmses):.4f} (+/- {np.std(rmses):.4f})")
    print(f"MAPE: {np.mean(mapes):.2f}% (+/- {np.std(mapes):.2f})")


def run_cv(engine, household_id: int, pool_limit: int, n_folds: int,
           min_train_hours: int, max_epochs: int, max_target_rows: int = 5000):
    all_ids = discover_real_households(engine, limit=pool_limit + 1)
    pool_household_ids = [h for h in all_ids if h != household_id][:pool_limit]
    print(f"Pooling {len(pool_household_ids)} OTHER households per fold (pool-limit={pool_limit}), "
          f"target household={household_id}")

    target_df = load_features(engine, household_id)
    cutoffs = generate_folds(len(target_df), HORIZON, n_folds, min_train_hours)
    print(f"Running {len(cutoffs)} fold(s), test_window_hours={HORIZON} (fixed - this is the 1-month model)")

    results = []

    for fold_i, cutoff in enumerate(cutoffs):
        print(f"\n===== Fold {fold_i + 1}/{len(cutoffs)} (cutoff hour={cutoff}) =====")
        keras.backend.clear_session()

        target_up_to_cutoff = target_df.iloc[:cutoff].copy()
        fold_df = target_df.iloc[:cutoff + HORIZON].copy()

        X_train, y_train, X_val, y_val = build_pooled_train_and_val_cv(
            engine, pool_household_ids, target_up_to_cutoff, max_target_rows=max_target_rows
        )
        print(f"Train sequences: {X_train.shape}, Val sequences: {X_val.shape}")

        model = build_bidirectional_direct_multistep(n_features=len(FEATURE_COLUMNS), horizon=HORIZON)
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=max_epochs, batch_size=16,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5, verbose=1),
            ],
            verbose=1,
        )

        del X_train, y_train, X_val, y_val
        gc.collect()

        result = fine_tune_and_forecast_1month_on_df(model, fold_df, household_id)
        print(f"Fold {fold_i + 1} - 1month: {result['metrics']}")
        results.append(result["metrics"])

    summarize(results, "1month (CV)")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--household-id", type=int, required=True,
                         help="The household to run rolling-origin CV against")
    parser.add_argument("--pool-limit", type=int, default=6)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--min-train-hours", type=int, default=3000,
                         help="Minimum history before the first fold's cutoff - each fold needs a full "
                              "720h test window on top of this")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--max-target-rows", type=int, default=5000)
    args = parser.parse_args()

    set_seeds(42)
    engine = create_engine(DATABASE_URL)
    run_cv(
        engine,
        household_id=args.household_id,
        pool_limit=args.pool_limit,
        n_folds=args.n_folds,
        min_train_hours=args.min_train_hours,
        max_epochs=args.max_epochs,
        max_target_rows=args.max_target_rows,
    )


if __name__ == "__main__":
    main()
