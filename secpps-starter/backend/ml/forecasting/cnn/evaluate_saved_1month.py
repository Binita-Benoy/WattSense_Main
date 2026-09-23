"""
Evaluate the SAVED 1-month direct multi-step model (no retraining).

For each household: the last 720 hours are held out as the test month, the
720 hours before that are the input window, and the forecast is compared
with the actual readings. The scaler is fitted on the pre-test data only,
the same way the CV script does it.

Households are split into two groups and reported separately:
  UNSEEN - not in the 15-household training pool -> honest, out-of-sample score
  SEEN   - in the training pool; the model trained on their last month,
           so these scores are optimistic (in-sample) and shown for reference only

A seasonal-naive baseline (repeat the last week of actual kWh) is scored on
the same months, so you can see whether the CNN beats a trivial forecast.
"""
import os
import argparse

import numpy as np
from sqlalchemy import create_engine
from tensorflow import keras

from features import load_features, scale_columns, compute_metrics, FEATURE_COLUMNS, discover_real_households
from direct_multistep_global import LOOKBACK, HORIZONS
from simple_cnn_lstm import mape

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)
HORIZON = HORIZONS["1month"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "global_direct_multistep_1month.keras")

# Copied from the training log ("Pooling 15 households ...")
TRAINED_IDS = {8162493, 8163939, 10042604, 10042756, 10042934, 10042974, 10174306, 10257896,
               10305996, 10341486, 10346126, 10346582, 10390745, 10413359, 10508173}


def evaluate_household(model, engine, hid):
    df = load_features(engine, hid)
    if len(df) < LOOKBACK + HORIZON:
        print(f"  {hid}: skipped, only {len(df)}h of data (need {LOOKBACK + HORIZON}h)")
        return None

    train_df, test_df = df.iloc[:-HORIZON], df.iloc[-HORIZON:]
    scaled_train, scalers = scale_columns(train_df)
    last_window = scaled_train[-LOOKBACK:, :].reshape(1, LOOKBACK, len(FEATURE_COLUMNS))

    scaled_forecast = model.predict(last_window, verbose=0).flatten()
    predicted = scalers["kwh"].inverse_transform(scaled_forecast.reshape(-1, 1)).flatten()
    actual = test_df["kwh"].values

    last_week = train_df["kwh"].values[-168:]
    naive = np.tile(last_week, HORIZON // 168 + 1)[:HORIZON]

    m = compute_metrics(actual, predicted)
    m["mape_corrected"] = round(mape(actual, predicted), 2)
    b = compute_metrics(actual, naive)
    b["mape_corrected"] = round(mape(actual, naive), 2)
    print(f"  {hid} (test starts {test_df.index[0]}): "
          f"CNN MAE={m['mae']} RMSE={m['rmse']} MAPEc={m['mape_corrected']} | "
          f"naive MAE={b['mae']} RMSE={b['rmse']} MAPEc={b['mape_corrected']}")
    return m, b


def summarise(label, results):
    if not results:
        print(f"\n=== {label}: no households evaluated ===")
        return
    print(f"\n=== {label} ({len(results)} households) ===")
    for key in ["mae", "rmse", "mape_corrected"]:
        cnn = np.mean([r[0][key] for r in results])
        base = np.mean([r[1][key] for r in results])
        print(f"{key:>15}:  CNN {cnn:.4f}   naive {base:.4f}   ({'CNN better' if cnn < base else 'naive better'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-unseen", type=int, default=5, help="How many unseen households to test")
    parser.add_argument("--skip-seen", action="store_true", help="Don't score the 15 training households")
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print(f"Loaded {MODEL_PATH}\nLOOKBACK={LOOKBACK}h, HORIZON={HORIZON}h\n")

    print("Finding households...", flush=True)
    all_ids = discover_real_households(engine, limit=len(TRAINED_IDS) + args.n_unseen + 5)
    print(f"Found {len(all_ids)} households", flush=True)    
    unseen_ids = [h for h in all_ids if h not in TRAINED_IDS][:args.n_unseen]

    print("UNSEEN households (out-of-sample):")
    unseen = [r for r in (evaluate_household(model, engine, h) for h in unseen_ids) if r]

    seen = []
    if not args.skip_seen:
        print("\nSEEN households (in training data -- optimistic):")
        seen = [r for r in (evaluate_household(model, engine, h) for h in sorted(TRAINED_IDS)) if r]

    summarise("UNSEEN - use these numbers", unseen)
    summarise("SEEN - reference only, in-sample", seen)
