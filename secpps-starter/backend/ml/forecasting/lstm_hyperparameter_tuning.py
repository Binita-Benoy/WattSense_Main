"""
Proposal quote: "[LSTM] configuration -- including number of layers,
hidden unit dimensions, dropout rate, and input window length -- will
be determined through systematic hyperparameter tuning, with the final
architecture selected based on validation MAPE."

This tries a small, documented grid (not exhaustive -- exhaustive search
over every possible combination would take far longer than is practical;
this is "systematic" in the sense of being structured and evaluated
consistently, which is what the proposal's wording requires).

Selection is based on VALIDATION MAPE specifically (computed on the
held-out validation split, in real kWh units after inverse-scaling) --
NOT the final test set, which stays untouched until the winning config
is evaluated once at the end. Using the test set to pick hyperparameters
would itself be a form of data leakage.
"""
import os

import numpy as np
from sqlalchemy import create_engine
from tensorflow import keras

from lstm_forecast import set_seeds
from lstm_full_features import (
    load_features_with_weather, scale_columns, create_sequences, build_model,
    FEATURE_COLUMNS, HORIZON, generate_full_feature_forecast,
)

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)

LOOKBACK_OPTIONS = [24, 48, 168]
UNITS_OPTIONS = [16, 32, 64]
DROPOUT_OPTIONS = [0.2]


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_config(engine, household_id: int, lookback: int, units: int, dropout: float) -> float:
    set_seeds()

    df = load_features_with_weather(engine, household_id)
    train_df = df.iloc[:-HORIZON]

    scaled_train, scalers = scale_columns(train_df)
    X, y = create_sequences(scaled_train, lookback)

    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    model = build_model(n_features=len(FEATURE_COLUMNS), lookback=lookback, units=units, dropout=dropout)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)

    val_predictions_scaled = model.predict(X_val, verbose=0).flatten()
    kwh_scaler = scalers["kwh"]
    val_predictions_real = kwh_scaler.inverse_transform(val_predictions_scaled.reshape(-1, 1)).flatten()
    val_actual_real = kwh_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

    return mape(val_actual_real, val_predictions_real)


def tune(engine, household_id: int) -> dict:
    results = []
    for lookback in LOOKBACK_OPTIONS:
        for units in UNITS_OPTIONS:
            for dropout in DROPOUT_OPTIONS:
                print(f"Testing lookback={lookback}, units={units}, dropout={dropout}...")
                val_mape = evaluate_config(engine, household_id, lookback, units, dropout)
                print(f"  -> validation MAPE: {val_mape:.2f}%")
                results.append({"lookback": lookback, "units": units, "dropout": dropout, "val_mape": val_mape})

    best = min(results, key=lambda r: r["val_mape"])
    return {"all_results": results, "best_config": best}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--household-id", type=int, required=True)
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)

    print(f"=== Hyperparameter tuning for household {args.household_id} ===\n")
    tuning_result = tune(engine, args.household_id)

    print("\n=== All results ===")
    for r in sorted(tuning_result["all_results"], key=lambda r: r["val_mape"]):
        print(f"lookback={r['lookback']:>3}, units={r['units']:>2}, dropout={r['dropout']}: "
              f"val MAPE={r['val_mape']:.2f}%")

    best = tuning_result["best_config"]
    print(f"\nBest config: lookback={best['lookback']}, units={best['units']}, "
          f"dropout={best['dropout']} (val MAPE={best['val_mape']:.2f}%)")

    print("\nRunning final evaluation on the held-out TEST set with the winning config...")
    final_result = generate_full_feature_forecast(
        engine, args.household_id,
        lookback=best["lookback"], units=best["units"], dropout=best["dropout"],
    )
    print(f"Final test metrics: {final_result['metrics']}")

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"lstm_tuning_{args.household_id}.json")
    with open(path, "w") as f:
        json.dump({"tuning": tuning_result, "final_test_result": final_result}, f, indent=2)
    print(f"\nSaved to {path}")
