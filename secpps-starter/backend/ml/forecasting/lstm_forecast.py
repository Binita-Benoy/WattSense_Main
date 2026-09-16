"""
US-05: LSTM energy forecast.

DESIGN DECISION, stated explicitly: this trains a SEPARATE LSTM per
household (same structure as arima_forecast.py / generate_weather_forecast.py),
for direct comparability in your model comparison table.

HONEST CAVEAT: per-household deep learning has a real overfitting risk --
LSTM has far more parameters than ARIMA, trained on one household's data
alone. This was a deliberate choice for consistency with the existing
per-household pipeline, not because it's guaranteed the best approach --
a pooled/global LSTM trained across multiple households (as your cited
literature actually does) would likely generalize better. Worth noting
as a limitation in your report, and worth revisiting if time allows.

Technical flow (matches the required task list exactly):
  Create sequences -> Scale data -> Train LSTM -> Predict -> Inverse scale -> Evaluate

Output format matches arima_forecast.py's result dict shape, so this can
slot into select_best_model.py / a future 3-way comparison script.

Requires: pip install tensorflow matplotlib (add to backend/requirements.txt)
NOTE: tensorflow is a large dependency (~500MB+) -- first docker build
after adding it will take noticeably longer than usual.
"""
import json
import os
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")  # no display available inside the container
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Seeded for practical reproducibility. NOTE: unlike ARIMA (no randomness
# at all), neural nets involve random weight initialization -- seeding
# gets you consistent, repeatable results on CPU, but this is a weaker
# reproducibility guarantee than ARIMA's exact determinism. Worth stating
# this distinction plainly rather than implying identical guarantees.
SEED = 42


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def load_series(engine, household_id: int) -> pd.Series:
    query = """
        SELECT reading_timestamp, consumption_kwh
        FROM energy_readings
        WHERE household_id = %(household_id)s
        ORDER BY reading_timestamp
    """
    df = pd.read_sql(query, engine, params={"household_id": household_id})
    if df.empty:
        raise ValueError(f"No readings found for household_id={household_id}")

    df["reading_timestamp"] = pd.to_datetime(df["reading_timestamp"])
    df = df.set_index("reading_timestamp")
    series = df["consumption_kwh"].resample("1h").mean().ffill().dropna()
    return series


def create_sequences(data: np.ndarray, lookback: int):
    """
    Converts a flat scaled series into supervised-learning format:
    each X is a window of `lookback` past hours, each y is the very
    next hour's value.
    """
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"mae": round(float(mae), 4), "rmse": round(float(rmse), 4), "mape": round(float(mape), 2)}


def build_model(lookback: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(lookback, 1)),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def generate_lstm_forecast(
    engine, household_id: int, horizon_hours: int = 24, lookback: int = 24,
    epochs: int = 50, batch_size: int = 32,
) -> dict:
    set_seeds()

    series = load_series(engine, household_id)
    if len(series) < horizon_hours + lookback + 100:
        raise ValueError(f"Not enough data for household {household_id}: {len(series)} hourly points")

    train_series = series.iloc[:-horizon_hours]
    test_series = series.iloc[-horizon_hours:]

    # ---- Scale data (RobustScaler, per your proposal's methodology) ----
    # Fit ONLY on train -- same leakage-avoidance principle as the
    # SARIMAX exog scaling: test data must stay genuinely unseen.
    scaler = RobustScaler()
    train_values = train_series.values.reshape(-1, 1)
    scaled_train = scaler.fit_transform(train_values).flatten()

    # ---- Create sequences ----
    X, y = create_sequences(scaled_train, lookback)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Time-respecting train/validation split -- last 10% chronologically,
    # NOT a random shuffle, since shuffling would leak future info into
    # validation for a time series.
    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # ---- Train LSTM ----
    model = build_model(lookback)
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stop], verbose=0,
    )

    final_train_loss = float(history.history["loss"][-1])
    final_val_loss = float(history.history["val_loss"][-1])
    print(f"Training complete after {len(history.history['loss'])} epochs "
          f"(early stopping may have triggered before {epochs})")
    print(f"Final train loss: {final_train_loss:.6f}, val loss: {final_val_loss:.6f}")

    # ---- Predict: recursive forecasting for the held-out horizon ----
    # Same approach as arima_forecast.py's evaluate=True mode: forecast
    # the real held-out hours, one step at a time, feeding each
    # prediction back in as input for the next step.
    window = scaled_train[-lookback:].copy()
    scaled_predictions = []
    for _ in range(horizon_hours):
        x_input = window.reshape((1, lookback, 1))
        next_scaled = model.predict(x_input, verbose=0)[0, 0]
        scaled_predictions.append(next_scaled)
        window = np.append(window[1:], next_scaled)

    # ---- Inverse scale ----
    scaled_predictions = np.array(scaled_predictions).reshape(-1, 1)
    predicted_kwh = scaler.inverse_transform(scaled_predictions).flatten()

    # Sanity check: predictions should be roughly in the same range as
    # real consumption, not still sitting in tiny scaled units -- this
    # is the actual verification behind "predictions are on the correct scale"
    actual_range = (train_series.min(), train_series.max())
    if not (actual_range[0] - 1 <= predicted_kwh.mean() <= actual_range[1] + 1):
        print(f"WARNING: predicted values (mean={predicted_kwh.mean():.3f}) look "
              f"out of range vs training data range {actual_range} -- check the inverse scaling")

    future_index = test_series.index
    predictions = [
        {"timestamp": ts.isoformat(), "predicted_kwh": round(float(v), 4)}
        for ts, v in zip(future_index, predicted_kwh)
    ]

    # ---- Evaluate ----
    metrics = compute_metrics(test_series.values, predicted_kwh)

    result = {
        "household_id": household_id,
        "model": "LSTM",
        "lookback_hours": lookback,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_hours": len(train_series),
        "horizon_hours": horizon_hours,
        "training": {
            "epochs_run": len(history.history["loss"]),
            "final_train_loss": round(final_train_loss, 6),
            "final_val_loss": round(final_val_loss, 6),
        },
        "predictions": predictions,
        "metrics": metrics,
        "actuals": [
            {"timestamp": ts.isoformat(), "actual_kwh": round(float(v), 4)}
            for ts, v in zip(test_series.index, test_series.values)
        ],
        "reproducibility_note": "Seeded (numpy, tensorflow) for practical repeatability on CPU -- weaker guarantee than ARIMA's exact determinism, since neural net training involves randomized weight initialization.",
    }

    return result, history


def plot_training_history(history, household_id: int) -> str:
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.title(f"LSTM training history — household {household_id}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss (scaled)")
    plt.legend()
    plt.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"lstm_training_history_{household_id}.png")
    plt.savefig(path)
    plt.close()
    return path


def plot_actual_vs_predicted(result: dict) -> str:
    timestamps = [p["timestamp"] for p in result["predictions"]]
    predicted = [p["predicted_kwh"] for p in result["predictions"]]
    actual = [a["actual_kwh"] for a in result["actuals"]]

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, actual, label="Actual", marker="o")
    plt.plot(timestamps, predicted, label="Predicted (LSTM)", marker="x")
    plt.title(f"Actual vs Predicted — household {result['household_id']}")
    plt.xlabel("Time")
    plt.ylabel("kWh")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"lstm_actual_vs_predicted_{result['household_id']}.png")
    plt.savefig(path)
    plt.close()
    return path


def save_forecast_result(result: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"lstm_{result['household_id']}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--household-id", type=int, required=True)
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)
    result, history = generate_lstm_forecast(
        engine, args.household_id, lookback=args.lookback, epochs=args.epochs
    )

    print(f"\nHousehold {result['household_id']}")
    print(f"Training: {result['training']}")
    print(f"Metrics: {result['metrics']}")
    print(f"First 3 predictions: {result['predictions'][:3]}")

    loss_plot_path = plot_training_history(history, args.household_id)
    forecast_plot_path = plot_actual_vs_predicted(result)
    json_path = save_forecast_result(result)

    print(f"\nTraining history plot saved to {loss_plot_path}")
    print(f"Actual vs predicted plot saved to {forecast_plot_path}")
    print(f"Full result saved to {json_path}")
