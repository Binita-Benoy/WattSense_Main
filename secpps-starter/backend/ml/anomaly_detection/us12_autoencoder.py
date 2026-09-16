import logging
logger = logging.getLogger()
# US-12: Multivariate LSTM Autoencoder for energy + weather anomaly detection.
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from config import (
    AE_SEQ_LEN, AE_BATCH_SIZE, AE_EPOCHS, AE_THRESHOLD_PCT, AE_PATIENCE,
    AE_MODEL_PATH, AE_SCALER_PKL, AE_THRESHOLD_PKL, AE_RESULTS_CSV, AE_GRAPH,
    COL_HOUSEHOLD, COL_KWH, OUT_GRAPHS
)

# Selected features are intentionally smaller than the full IF set. They represent
# complementary temporal, consumption and weather signals for sequence reconstruction.
AE_FEATURES = [
    COL_KWH,
    "lag_1h", "lag_24h", "lag_168h",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "rolling_24h_mean", "rolling_24h_std", "deviation_24h", "z_score",
    "change_1h", "peak_ratio",
    "temperature", "apparent_temperature", "humidity", "wind_speed",
    "precip_intensity", "dew_point", "shortwave_radiation",
    "temperature_deviation_24h", "cooling_degree", "heating_degree"
]


def make_sequences(values: np.ndarray, seq_len=AE_SEQ_LEN):
    X, indices = [], []
    for i in range(len(values) - seq_len + 1):
        X.append(values[i:i + seq_len])
        indices.append(i + seq_len - 1)
    return np.asarray(X, dtype=np.float32), indices


def build_autoencoder(seq_len, n_features):
    try:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Dropout
    except ImportError:
        raise ImportError("TensorFlow not found. Run: pip install tensorflow")

    inp = Input(shape=(seq_len, n_features))
    x = LSTM(64, activation="relu", return_sequences=True)(inp)
    x = Dropout(0.2)(x)
    x = LSTM(32, activation="relu", return_sequences=False)(x)
    x = RepeatVector(seq_len)(x)
    x = LSTM(32, activation="relu", return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(64, activation="relu", return_sequences=True)(x)
    out = TimeDistributed(Dense(n_features))(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model


def _usable_features(F_train):
    available = [c for c in AE_FEATURES if c in F_train.columns]
    missing = [c for c in AE_FEATURES if c not in F_train.columns]
    if missing:
        logger.info(f"AE features unavailable/skipped: {missing}")
    return available


def prepare_split_sequences(F_train, F_val, F_test):
    logger.info("\n" + "=" * 70)
    logger.info("US-12: PREPARING MULTIVARIATE LSTM SEQUENCES")
    logger.info("=" * 70)

    features = _usable_features(F_train)
    scaler = MinMaxScaler()
    train_values = F_train[features].astype(float).replace([np.inf, -np.inf], np.nan)
    train_values = train_values.fillna(train_values.median())
    scaler.fit(train_values)
    joblib.dump(scaler, AE_SCALER_PKL)

    results = {"train": {"X": [], "meta": []},
               "val": {"X": [], "meta": []},
               "test": {"X": [], "meta": []},
               "scaler": scaler, "feature_cols": features}

    households = sorted(F_train[COL_HOUSEHOLD].unique())
    for hh_id in households:
        for split_name, F_split in [("train", F_train), ("val", F_val), ("test", F_test)]:
            df_hh = F_split[F_split[COL_HOUSEHOLD] == hh_id].sort_index().copy()
            if len(df_hh) < AE_SEQ_LEN:
                continue
            vals = df_hh[features].astype(float).replace([np.inf, -np.inf], np.nan)
            vals = vals.fillna(train_values.median())
            scaled = scaler.transform(vals)
            X, idx = make_sequences(scaled, AE_SEQ_LEN)
            results[split_name]["X"].append(X)
            ts = df_hh.index
            for j in idx:
                results[split_name]["meta"].append({
                    "timestamp": ts[j],
                    COL_HOUSEHOLD: hh_id,
                    "consumption": float(df_hh.iloc[j][COL_KWH])
                })

    for split in ["train", "val", "test"]:
        if results[split]["X"]:
            results[split]["X"] = np.concatenate(results[split]["X"], axis=0)
            results[split]["meta"] = pd.DataFrame(results[split]["meta"])
        else:
            results[split]["X"] = np.empty((0, AE_SEQ_LEN, len(features)), dtype=np.float32)
            results[split]["meta"] = pd.DataFrame(columns=["timestamp", COL_HOUSEHOLD, "consumption"])
        logger.info(f"{split.upper()} sequences: {len(results[split]['X']):,}")

    logger.info(f"Features used by AE ({len(features)}): {features}")
    return results


def train_autoencoder(seq_data):
    try:
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    except ImportError:
        raise ImportError("TensorFlow not found. Run: pip install tensorflow")

    X_train = seq_data["train"]["X"]
    X_val = seq_data["val"]["X"]
    model = build_autoencoder(X_train.shape[1], X_train.shape[2])

    history = model.fit(
        X_train, X_train,
        epochs=AE_EPOCHS,
        batch_size=AE_BATCH_SIZE,
        validation_data=(X_val, X_val),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=AE_PATIENCE,
                          restore_best_weights=True, verbose=0),
            ModelCheckpoint(AE_MODEL_PATH, save_best_only=True, verbose=0)
        ],
        shuffle=True,
        verbose=0
    )
    best = int(np.argmin(history.history["val_loss"]) + 1)
    logger.info(f"Training complete. Best epoch: {best}")
    logger.info(f"Final train loss: {history.history['loss'][-1]:.6f}")
    logger.info(f"Final val loss: {history.history['val_loss'][-1]:.6f}")
    _plot_training(history)
    return model


def _plot_training(history):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], label="Train loss")
    ax.plot(history.history["val_loss"], label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("US-12 Multivariate LSTM Autoencoder — Training Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(OUT_GRAPHS, "us12_training_loss.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def reconstruction_errors(model, X):
    pred = model.predict(X, batch_size=256, verbose=0)
    return np.mean(np.abs(X - pred), axis=(1, 2))


def compute_threshold(model, X_train):
    errors = reconstruction_errors(model, X_train)
    threshold = float(np.percentile(errors, AE_THRESHOLD_PCT))
    joblib.dump(threshold, AE_THRESHOLD_PKL)
    logger.info(f"Train reconstruction error mean: {errors.mean():.6f}")
    logger.info(f"Train reconstruction threshold ({AE_THRESHOLD_PCT}th): {threshold:.6f}")
    return threshold


def predict_on_split(model, X_split, metadata, threshold, split_name="test"):
    if len(X_split) == 0:
        return pd.DataFrame()
    errors = reconstruction_errors(model, X_split)
    score = np.clip((errors - errors.min()) /
                    (errors.max() - errors.min() + 1e-9), 0, 1)
    result = metadata.copy()
    result["recon_error"] = errors
    result["threshold"] = threshold
    result["anomaly_score"] = score
    result["is_anomaly"] = (errors > threshold).astype(int)
    result["split"] = split_name
    return result


def plot_ae_evidence(results):
    if results.empty:
        return
    hh = results[COL_HOUSEHOLD].iloc[0]
    df = results[results[COL_HOUSEHOLD] == hh].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    normal = df[df["is_anomaly"] == 0]
    anom = df[df["is_anomaly"] == 1]
    axes[0].plot(normal["timestamp"], normal["consumption"], lw=0.6, label="Normal")
    axes[0].scatter(anom["timestamp"], anom["consumption"], s=18, label="Anomaly")
    axes[0].set_ylabel("kWh")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(df["timestamp"], df["recon_error"], lw=0.7, label="Reconstruction error")
    axes[1].axhline(df["threshold"].iloc[0], ls="--", lw=1, label="Threshold")
    axes[1].set_ylabel("MAE")
    axes[1].set_xlabel("Timestamp")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(AE_GRAPH, dpi=150, bbox_inches="tight")
    plt.close()


def run_ae_pipeline(splits):
    seq_data = prepare_split_sequences(splits["train"], splits["val"], splits["test"])
    model = train_autoencoder(seq_data)
    threshold = compute_threshold(model, seq_data["train"]["X"])
    results_val = predict_on_split(model, seq_data["val"]["X"], seq_data["val"]["meta"], threshold, "val")
    results_test = predict_on_split(model, seq_data["test"]["X"], seq_data["test"]["meta"], threshold, "test")
    results_test.to_csv(AE_RESULTS_CSV, index=False)
    logger.info(f"Test AE anomalies: {results_test['is_anomaly'].sum():,} / {len(results_test):,}")
    logger.info(f"Saved: {AE_RESULTS_CSV}")
    plot_ae_evidence(results_test)
    return {
        "model": model,
        "threshold": threshold,
        "scaler": seq_data["scaler"],
        "feature_cols": seq_data["feature_cols"],
        "results_val": results_val,
        "results_test": results_test,
        "sequence_data": seq_data
    }
