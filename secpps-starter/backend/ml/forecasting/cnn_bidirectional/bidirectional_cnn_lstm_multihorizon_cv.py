"""
5-fold rolling-origin CV for the Bidirectional LSTM variant's 24h/7-day
horizons - same CV pattern as both earlier variants' CV scripts
(timedistributed_cnn_lstm_multihorizon_cv.py, direct_multistep_1month_cv.py,
simple_cnn_lstm_cv.py): each fold retrains the full pooled base model using
all other households' complete history plus the target household's data up
to that fold's cutoff, then fine-tunes and forecasts forward into that
fold's held-out window. Nothing from a fold's test window is ever seen
during that fold's training.

Memory fixes applied from the start, not discovered via a crash this time:
the recursive TimeDistributed CV script died with a silent, no-traceback
OOM on its first run (pooled training arrays still held in memory when
fine-tuning's clone_model() duplicated the model right after). This script's
LOOKBACK is only 24 (vs 168 for TimeDistributed), so sequences here are
~7x smaller and the same crash is less likely - but the fix costs nothing
to include and there's no reason to wait for a crash to add it a third time.

Efficiency note: one 168h recursive run per fold serves both the 24h result
(its first 24 hours) and the 7-day result, same trick as the other variants.

Usage (GPU-first, CPU-fallback-only-if-it-fails, per your standing instruction):
    docker compose exec -T backend python ml/forecasting/cnn_bidirectional/bidirectional_cnn_lstm_multihorizon_cv.py --household-id 8162493 --pool-limit 6
    docker compose exec -T -e CUDA_VISIBLE_DEVICES="" backend python ml/forecasting/cnn_bidirectional/bidirectional_cnn_lstm_multihorizon_cv.py --household-id 8162493 --pool-limit 6
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
    create_sequences,
    discover_real_households,
    set_seeds,
)
from simple_cnn_lstm import mape
from bidirectional_cnn_lstm_multihorizon import (
    LOOKBACK,
    build_bidirectional_cnn_lstm,
    apply_scalers,
    _fine_tune_and_forecast_recursive_on_df,
)

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)


def generate_folds(total_hours: int, test_window_hours: int, n_folds: int, min_train_hours: int) -> list:
    """Same generic rolling-origin cutoff generator used by every CV script
    in this project - evenly-spaced cutoffs, each leaving test_window_hours
    of real future data after it, auto-reducing n_folds (with a warning,
    never a silent truncation) if history can't fit the request."""
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
    """Same as build_pooled_train_and_val in the single-split script, plus
    the target household's own up-to-cutoff history folded in as EXTRA
    TRAIN-ONLY data, capped at max_target_rows (most recent rows kept) so
    later folds don't use progressively more memory than earlier ones."""
    X_train_all, y_train_all = [], []
    X_val_all, y_val_all = [], []

    if len(target_df_up_to_cutoff) > max_target_rows:
        target_df_up_to_cutoff = target_df_up_to_cutoff.iloc[-max_target_rows:]

    for hh_id in pool_household_ids:
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

    if len(target_df_up_to_cutoff) >= LOOKBACK + 48:
        tgt_scaled, _ = scale_columns(target_df_up_to_cutoff)
        X_tgt, y_tgt = create_sequences(tgt_scaled, LOOKBACK)
        if len(X_tgt) > 0:
            X_train_all.append(X_tgt)
            y_train_all.append(y_tgt)

    X_train = np.concatenate(X_train_all).astype("float32")
    y_train = np.concatenate(y_train_all).astype("float32")
    X_val = np.concatenate(X_val_all).astype("float32")
    y_val = np.concatenate(y_val_all).astype("float32")
    return X_train, y_train, X_val, y_val


def metrics_from_predictions(predictions: list) -> dict:
    y_pred = np.array([p["predicted_kwh"] for p in predictions])
    y_true = np.array([p["actual_kwh"] for p in predictions])
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mape": round(float(mape(y_true, y_pred)), 2),
    }


def summarize(results: list, label: str):
    maes = [r["mae"] for r in results]
    rmses = [r["rmse"] for r in results]
    mapes = [r["mape"] for r in results]
    print(f"\n=== {label} ({len(results)} fold{'s' if len(results) != 1 else ''}) ===")
    print(f"MAE:  {np.mean(maes):.4f} (+/- {np.std(maes):.4f})")
    print(f"RMSE: {np.mean(rmses):.4f} (+/- {np.std(rmses):.4f})")
    print(f"MAPE: {np.mean(mapes):.2f}% (+/- {np.std(mapes):.2f})")


def run_cv(engine, household_id: int, pool_limit: int, n_folds: int,
           test_window_hours: int, min_train_hours: int, max_epochs: int,
           max_target_rows: int = 5000):
    all_ids = discover_real_households(engine, limit=pool_limit + 1)
    pool_household_ids = [h for h in all_ids if h != household_id][:pool_limit]
    print(f"Pooling {len(pool_household_ids)} OTHER households per fold (pool-limit={pool_limit}), "
          f"target household={household_id}")

    target_df = load_features(engine, household_id)
    cutoffs = generate_folds(len(target_df), test_window_hours, n_folds, min_train_hours)
    print(f"Running {len(cutoffs)} fold(s), test_window_hours={test_window_hours}")

    results_24h, results_7day = [], []

    for fold_i, cutoff in enumerate(cutoffs):
        print(f"\n===== Fold {fold_i + 1}/{len(cutoffs)} (cutoff hour={cutoff}) =====")
        keras.backend.clear_session()

        target_up_to_cutoff = target_df.iloc[:cutoff].copy()
        fold_df = target_df.iloc[:cutoff + test_window_hours].copy()

        X_train, y_train, X_val, y_val = build_pooled_train_and_val_cv(
            engine, pool_household_ids, target_up_to_cutoff, max_target_rows=max_target_rows
        )
        print(f"Train sequences: {X_train.shape}, Val sequences: {X_val.shape}")

        model = build_bidirectional_cnn_lstm(n_features=len(FEATURE_COLUMNS))
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=max_epochs, batch_size=32,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5, verbose=1),
            ],
            verbose=1,
        )

        # Memory fix applied preemptively (see module docstring) rather than
        # waiting for the same silent OOM that hit the TimeDistributed CV
        # script on its first run.
        del X_train, y_train, X_val, y_val
        gc.collect()

        result_7day = _fine_tune_and_forecast_recursive_on_df(
            model, fold_df, household_id, horizon_hours=test_window_hours
        )
        metrics_7day = result_7day["metrics"]
        metrics_24h = metrics_from_predictions(result_7day["predictions"][:24])

        print(f"Fold {fold_i + 1} - 24h:  {metrics_24h}")
        print(f"Fold {fold_i + 1} - 7day: {metrics_7day}")

        results_24h.append(metrics_24h)
        results_7day.append(metrics_7day)

    summarize(results_24h, "24h (CV)")
    summarize(results_7day, "7day (CV)")
    return results_24h, results_7day


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--household-id", type=int, required=True,
                         help="The household to run rolling-origin CV against")
    parser.add_argument("--pool-limit", type=int, default=6,
                         help="Other households pooled per fold")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--test-window-hours", type=int, default=168,
                         help="Held-out window per fold - 168h also covers the 24h result as its first slice")
    parser.add_argument("--min-train-hours", type=int, default=2000,
                         help="Minimum history before the first fold's cutoff")
    parser.add_argument("--max-epochs", type=int, default=20,
                         help="Lower this for a faster first look before committing to the full run")
    parser.add_argument("--max-target-rows", type=int, default=5000,
                         help="Cap on target household's own up-to-cutoff rows fed into pooling per fold")
    args = parser.parse_args()

    set_seeds(42)
    engine = create_engine(DATABASE_URL)
    run_cv(
        engine,
        household_id=args.household_id,
        pool_limit=args.pool_limit,
        n_folds=args.n_folds,
        test_window_hours=args.test_window_hours,
        min_train_hours=args.min_train_hours,
        max_epochs=args.max_epochs,
        max_target_rows=args.max_target_rows,
    )


if __name__ == "__main__":
    main()
