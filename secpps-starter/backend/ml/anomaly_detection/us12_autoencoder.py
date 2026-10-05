"""US-12: Multivariate LSTM Autoencoder with point-aligned anomaly scoring."""
from __future__ import annotations

import logging
import os
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import (
    AE_SEQ_LEN,
    AE_MAX_TRAIN_SEQUENCES,
    AE_MAX_VAL_SEQUENCES,
    AE_MAX_TEST_SEQUENCES,
    AE_BATCH_SIZE,
    AE_EPOCHS,
    AE_THRESHOLD_PCT,
    AE_PATIENCE,
    AE_LSTM_UNITS_1,
    AE_LSTM_UNITS_2,
    AE_PREDICT_BATCH_SIZE,
    AE_RANDOM_STATE,
    AE_SCORE_LOW_PCT,
    AE_SCORE_HIGH_PCT,
    AE_MODEL_PATH,
    AE_SCALER_PKL,
    AE_PREPROCESS_PKL,
    AE_THRESHOLD_PKL,
    AE_RESULTS_CSV,
    AE_GRAPH,
    COL_HOUSEHOLD,
    COL_KWH,
    COL_PUBLIC_HOLIDAY,
    OUT_GRAPHS,
)
from logging_utils import log_print, safe_savefig

logger = logging.getLogger()

AE_FEATURES = [
    COL_KWH,
    # Historical energy context.
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_24h_mean", "rolling_24h_std",
    "deviation_24h", "z_score", "change_1h", "peak_ratio", "lag24_ratio",
    # Calendar context.
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", COL_PUBLIC_HOLIDAY,
    # Weather context.
    "temperature", "apparent_temperature", "humidity", "wind_speed",
    "precip_intensity", "dew_point", "shortwave_radiation",
    "temperature_deviation_24h", "cooling_degree", "heating_degree",
]

# Reconstruction is still multivariate, but the anomaly decision focuses on the
# current timestep's energy-sensitive dimensions. This prevents one point anomaly
# from being diluted across 24 x N reconstructed values.
AE_SCORE_FEATURES = [
    COL_KWH,
    "deviation_24h",
    "z_score",
    "change_1h",
    "peak_ratio",
    "lag24_ratio",
]


def make_sequences(values: np.ndarray, seq_len: int = AE_SEQ_LEN):
    """Convert a 2-D [time, features] matrix to overlapping sequences."""
    if values.ndim != 2:
        raise ValueError("values must have shape [time, features]")
    n = len(values) - seq_len + 1
    if n <= 0:
        return (
            np.empty((0, seq_len, values.shape[1]), dtype=np.float32),
            [],
        )
    X = np.asarray(
        [values[i : i + seq_len] for i in range(n)],
        dtype=np.float32,
    )
    endpoints = list(range(seq_len - 1, len(values)))
    return X, endpoints


def _sample_sequences(X, meta, max_sequences, seed):
    """Deterministically cap sequences to keep laptop RAM/training manageable."""
    if max_sequences is None or len(X) <= max_sequences:
        return X, meta.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(X), size=int(max_sequences), replace=False))
    return X[keep], meta.iloc[keep].reset_index(drop=True)


def _sample_per_household(X, meta, max_sequences, seed):
    """Cap total sequences while retaining representation from every household."""
    if max_sequences is None or len(X) <= max_sequences:
        return X, meta.reset_index(drop=True)

    households = sorted(meta[COL_HOUSEHOLD].unique())
    per_hh = max(1, int(max_sequences) // max(1, len(households)))
    rng = np.random.default_rng(seed)
    keep_parts = []

    hh_values = meta[COL_HOUSEHOLD].to_numpy()
    for hh in households:
        idx = np.flatnonzero(hh_values == hh)
        if len(idx) > per_hh:
            idx = rng.choice(idx, size=per_hh, replace=False)
        keep_parts.append(np.sort(idx))

    keep = np.concatenate(keep_parts)
    if len(keep) > max_sequences:
        keep = np.sort(rng.choice(keep, size=max_sequences, replace=False))
    else:
        keep = np.sort(keep)
    return X[keep], meta.iloc[keep].reset_index(drop=True)


def build_autoencoder(seq_len: int, n_features: int):
    """Build the multivariate sequence-to-sequence LSTM autoencoder."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        from tensorflow.keras.layers import (
            Dense,
            Dropout,
            Input,
            LSTM,
            RepeatVector,
            TimeDistributed,
        )
        from tensorflow.keras.models import Model
    except ImportError as exc:
        raise ImportError("TensorFlow not found. Run: pip install tensorflow") from exc

    inp = Input(shape=(seq_len, n_features))
    x = LSTM(AE_LSTM_UNITS_1, activation="tanh", return_sequences=True)(inp)
    x = Dropout(0.15)(x)
    x = LSTM(AE_LSTM_UNITS_2, activation="tanh", return_sequences=False)(x)
    x = RepeatVector(seq_len)(x)
    x = LSTM(AE_LSTM_UNITS_2, activation="tanh", return_sequences=True)(x)
    x = Dropout(0.15)(x)
    x = LSTM(AE_LSTM_UNITS_1, activation="tanh", return_sequences=True)(x)
    out = TimeDistributed(Dense(n_features))(x)

    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model


def _usable_features(F_train: pd.DataFrame) -> list[str]:
    available = [c for c in AE_FEATURES if c in F_train.columns]
    missing = [c for c in AE_FEATURES if c not in F_train.columns]
    if missing:
        log_print(f"AE features unavailable/skipped: {missing}")
    if COL_KWH not in available:
        raise ValueError(f"Required AE feature '{COL_KWH}' is missing.")
    return available


def _prepare_numeric(df: pd.DataFrame, features: list[str], medians: pd.Series) -> pd.DataFrame:
    values = df[features].astype(float).replace([np.inf, -np.inf], np.nan)
    return values.fillna(medians)


def _build_household_sequences(
    target_hh: pd.DataFrame,
    history_hh: pd.DataFrame | None,
    features: list[str],
    train_medians: pd.Series,
    scaler: MinMaxScaler,
):
    """
    Build sequences whose endpoints belong to target_hh.

    Up to seq_len-1 historical rows are prepended so the first validation/test
    reading can use context from the immediately preceding split.
    """
    target_hh = target_hh.sort_index().copy()
    if target_hh.empty:
        return np.empty((0, AE_SEQ_LEN, len(features)), dtype=np.float32), pd.DataFrame()

    history_hh = (
        history_hh.sort_index().tail(AE_SEQ_LEN - 1).copy()
        if history_hh is not None and not history_hh.empty
        else target_hh.iloc[0:0].copy()
    )

    combined = pd.concat([history_hh, target_hh], axis=0)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    values = _prepare_numeric(combined, features, train_medians)
    scaled = scaler.transform(values)
    X_all, endpoints = make_sequences(scaled, AE_SEQ_LEN)
    if len(X_all) == 0:
        return X_all, pd.DataFrame()

    target_index = set(target_hh.index)
    keep_positions = [
        i for i, endpoint in enumerate(endpoints)
        if combined.index[endpoint] in target_index
    ]
    if not keep_positions:
        return np.empty((0, AE_SEQ_LEN, len(features)), dtype=np.float32), pd.DataFrame()

    X = X_all[keep_positions]
    selected_endpoints = [endpoints[i] for i in keep_positions]
    metadata = pd.DataFrame(
        {
            "timestamp": [combined.index[j] for j in selected_endpoints],
            COL_HOUSEHOLD: int(target_hh[COL_HOUSEHOLD].iloc[0]),
            "consumption": [float(combined.iloc[j][COL_KWH]) for j in selected_endpoints],
        }
    )
    return X, metadata


def build_sequences_for_split(
    F_target: pd.DataFrame,
    F_history: pd.DataFrame | None,
    preprocessing: dict,
    max_sequences: int | None = None,
    seed: int = AE_RANDOM_STATE,
):
    """Build context-aware sequences for an arbitrary validation/test DataFrame."""
    features = preprocessing["feature_cols"]
    medians = preprocessing["train_medians"]
    scaler = preprocessing["scaler"]

    X_parts = []
    meta_parts = []
    for hh_id in sorted(F_target[COL_HOUSEHOLD].unique()):
        target_hh = F_target[F_target[COL_HOUSEHOLD] == hh_id].copy()
        history_hh = None
        if F_history is not None and not F_history.empty:
            history_hh = F_history[F_history[COL_HOUSEHOLD] == hh_id].copy()

        X_hh, meta_hh = _build_household_sequences(
            target_hh, history_hh, features, medians, scaler
        )
        if len(X_hh):
            X_parts.append(X_hh)
            meta_parts.append(meta_hh)

    if not X_parts:
        return (
            np.empty((0, AE_SEQ_LEN, len(features)), dtype=np.float32),
            pd.DataFrame(columns=["timestamp", COL_HOUSEHOLD, "consumption"]),
        )

    X = np.concatenate(X_parts, axis=0).astype(np.float32, copy=False)
    meta = pd.concat(meta_parts, ignore_index=True)
    X, meta = _sample_per_household(X, meta, max_sequences, seed)
    return X, meta


def prepare_split_sequences(F_train, F_val, F_test):
    """Fit preprocessing on TRAIN only and build context-aware sequences."""
    log_print("\n" + "=" * 70)
    log_print("US-12: PREPARING MULTIVARIATE LSTM SEQUENCES")
    log_print("=" * 70)

    features = _usable_features(F_train)
    train_values = F_train[features].astype(float).replace([np.inf, -np.inf], np.nan)
    train_medians = train_values.median().fillna(0.0)
    train_values = train_values.fillna(train_medians)

    scaler = MinMaxScaler()
    scaler.fit(train_values)
    joblib.dump(scaler, AE_SCALER_PKL)

    preprocessing = {
        "feature_cols": features,
        "train_medians": train_medians,
        "scaler": scaler,
    }

    # Validation can use the tail of train as legitimate historical context;
    # test can use the tail of train+validation.
    train_history = F_train.iloc[0:0].copy()
    val_history = F_train
    test_history = pd.concat([F_train, F_val], axis=0)

    split_specs = {
        "train": (F_train, train_history, AE_MAX_TRAIN_SEQUENCES, AE_RANDOM_STATE + 1),
        "val": (F_val, val_history, AE_MAX_VAL_SEQUENCES, AE_RANDOM_STATE + 2),
        "test": (F_test, test_history, AE_MAX_TEST_SEQUENCES, AE_RANDOM_STATE + 3),
    }

    results = {
        "scaler": scaler,
        "feature_cols": features,
        "train_medians": train_medians,
    }

    for name, (target, history, cap, seed) in split_specs.items():
        X, meta = build_sequences_for_split(
            target,
            history,
            preprocessing,
            max_sequences=cap,
            seed=seed,
        )
        results[name] = {"X": X, "meta": meta}
        log_print(
            f"{name.upper()} sequences: {len(X):,} | "
            f"memory: {X.nbytes / (1024 ** 2):.1f} MB"
        )

    score_features = [c for c in AE_SCORE_FEATURES if c in features]
    score_indices = [features.index(c) for c in score_features]
    if not score_indices:
        score_features = [COL_KWH]
        score_indices = [features.index(COL_KWH)]

    results["score_features"] = score_features
    results["score_indices"] = score_indices

    joblib.dump(
        {
            "feature_cols": features,
            "train_medians": train_medians,
            "score_features": score_features,
            "score_indices": score_indices,
        },
        AE_PREPROCESS_PKL,
    )

    log_print(f"AE features ({len(features)}): {features}")
    log_print(f"AE anomaly-score features: {score_features}")
    log_print(
        f"Laptop settings: max train={AE_MAX_TRAIN_SEQUENCES:,}, "
        f"max val={AE_MAX_VAL_SEQUENCES:,}, max test={AE_MAX_TEST_SEQUENCES:,}, "
        f"batch={AE_BATCH_SIZE}, epochs={AE_EPOCHS}, patience={AE_PATIENCE}"
    )
    return results


def train_autoencoder(seq_data):
    """Train the LSTM autoencoder using chronological validation sequences."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
    except ImportError as exc:
        raise ImportError("TensorFlow not found. Run: pip install tensorflow") from exc

    random.seed(AE_RANDOM_STATE)
    np.random.seed(AE_RANDOM_STATE)
    tf.random.set_seed(AE_RANDOM_STATE)

    X_train = seq_data["train"]["X"]
    X_val = seq_data["val"]["X"]
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("US-12 requires non-empty training and validation sequences.")

    model = build_autoencoder(X_train.shape[1], X_train.shape[2])

    class EpochLogger(Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            log_print(
                f"Epoch {epoch + 1:02d}/{AE_EPOCHS} | "
                f"loss={logs.get('loss', float('nan')):.6f} | "
                f"val_loss={logs.get('val_loss', float('nan')):.6f}"
            )

    log_print("\nStarting LSTM autoencoder training...")
    log_print(f"Training samples: {len(X_train):,}")
    log_print(f"Validation samples: {len(X_val):,}")
    log_print(f"Input shape: {X_train.shape[1:]}")

    history = model.fit(
        X_train,
        X_train,
        epochs=AE_EPOCHS,
        batch_size=AE_BATCH_SIZE,
        validation_data=(X_val, X_val),
        callbacks=[
            EpochLogger(),
            EarlyStopping(
                monitor="val_loss",
                patience=AE_PATIENCE,
                restore_best_weights=True,
                verbose=0,
            ),
            ModelCheckpoint(AE_MODEL_PATH, save_best_only=True, verbose=0),
        ],
        shuffle=True,
        verbose=0,
    )

    best = int(np.argmin(history.history["val_loss"]) + 1)
    log_print(f"Training complete. Best epoch: {best}")
    log_print(f"Best val loss: {min(history.history['val_loss']):.6f}")
    _plot_training(history)
    return model


def _plot_training(history):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], label="Train loss")
    ax.plot(history.history["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("US-12 Multivariate LSTM Autoencoder — Training Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(OUT_GRAPHS, "us12_training_loss.png")
    safe_savefig(fig, path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log_print(f"Training curve saved: {path}")


def reconstruction_errors(model, X: np.ndarray, score_indices: list[int]) -> np.ndarray:
    """
    Score the CURRENT/final timestep only, over energy-sensitive dimensions.

    The complete 24-hour multivariate sequence is still reconstructed, so weather
    and temporal context influence the latent representation. The point anomaly
    score is aligned with the sequence endpoint used by US-13 ground truth.
    """
    if len(X) == 0:
        return np.empty(0, dtype=np.float32)

    pred = model.predict(X, batch_size=AE_PREDICT_BATCH_SIZE, verbose=0)
    actual_current = X[:, -1, score_indices]
    predicted_current = pred[:, -1, score_indices]
    return np.mean(np.abs(actual_current - predicted_current), axis=1)


def _normalise_errors(errors: np.ndarray, low: float, high: float) -> np.ndarray:
    denom = max(float(high) - float(low), 1e-9)
    return np.clip((errors - float(low)) / denom, 0.0, 1.0)


def compute_error_reference(model, seq_data):
    """Establish fallback threshold + stable train-referenced score scale."""
    errors = reconstruction_errors(
        model,
        seq_data["train"]["X"],
        seq_data["score_indices"],
    )
    if len(errors) == 0:
        raise ValueError("No AE training errors available.")

    threshold = float(np.percentile(errors, AE_THRESHOLD_PCT))
    score_low = float(np.percentile(errors, AE_SCORE_LOW_PCT))
    score_high = float(np.percentile(errors, AE_SCORE_HIGH_PCT))

    seq_data["threshold"] = threshold
    seq_data["threshold_source"] = f"train_{AE_THRESHOLD_PCT:g}th_percentile_fallback"
    seq_data["score_low"] = score_low
    seq_data["score_high"] = score_high

    joblib.dump(threshold, AE_THRESHOLD_PKL)
    _save_preprocessing(seq_data)

    log_print(f"Train current-step reconstruction error mean: {errors.mean():.6f}")
    log_print(f"Fallback AE threshold ({AE_THRESHOLD_PCT:g}th): {threshold:.6f}")
    return threshold


def _save_preprocessing(seq_data):
    payload = {
        "feature_cols": seq_data["feature_cols"],
        "train_medians": seq_data["train_medians"],
        "score_features": seq_data["score_features"],
        "score_indices": seq_data["score_indices"],
        "score_low": seq_data.get("score_low"),
        "score_high": seq_data.get("score_high"),
        "threshold": seq_data.get("threshold"),
        "threshold_source": seq_data.get("threshold_source"),
    }
    joblib.dump(payload, AE_PREPROCESS_PKL)


def set_ae_threshold(seq_data, threshold: float, source: str = "validation_f1"):
    seq_data["threshold"] = float(threshold)
    seq_data["threshold_source"] = source
    joblib.dump(float(threshold), AE_THRESHOLD_PKL)
    _save_preprocessing(seq_data)
    log_print(f"AE threshold set to {threshold:.6f} ({source})")
    return seq_data


def score_ae_sequences(model, X: np.ndarray, seq_data) -> pd.DataFrame:
    errors = reconstruction_errors(model, X, seq_data["score_indices"])
    if len(errors) == 0:
        return pd.DataFrame(columns=["recon_error", "anomaly_score"])
    scores = _normalise_errors(
        errors,
        seq_data["score_low"],
        seq_data["score_high"],
    )
    return pd.DataFrame({"recon_error": errors, "anomaly_score": scores})


def predict_on_split(
    model,
    X_split,
    metadata,
    seq_data,
    split_name: str = "test",
    threshold: float | None = None,
):
    if len(X_split) == 0:
        return pd.DataFrame()

    threshold = float(seq_data["threshold"] if threshold is None else threshold)
    scored = score_ae_sequences(model, X_split, seq_data)

    result = metadata.reset_index(drop=True).copy()
    result["recon_error"] = scored["recon_error"].to_numpy()
    result["anomaly_score"] = scored["anomaly_score"].to_numpy()
    result["threshold"] = threshold
    result["is_anomaly"] = (result["recon_error"] >= threshold).astype("int8")
    result["split"] = split_name
    return result


def plot_ae_evidence(results: pd.DataFrame):
    if results.empty:
        return

    hh = int(results[COL_HOUSEHOLD].iloc[0])
    df = results[results[COL_HOUSEHOLD] == hh].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    normal = df[df["is_anomaly"] == 0]
    anomalies = df[df["is_anomaly"] == 1]

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    fig.suptitle(f"US-12 LSTM Autoencoder — Household {hh}", fontsize=12)

    axes[0].plot(df["timestamp"], df["consumption"], lw=0.55, label="Consumption")
    axes[0].scatter(
        anomalies["timestamp"],
        anomalies["consumption"],
        s=20,
        zorder=5,
        label=f"Detected anomaly ({len(anomalies)})",
    )
    axes[0].set_ylabel("kWh")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(df["timestamp"], df["recon_error"], lw=0.7, label="Current-step reconstruction error")
    axes[1].axhline(df["threshold"].iloc[0], ls="--", lw=1.2, label="Detection threshold")
    axes[1].scatter(
        anomalies["timestamp"],
        anomalies["recon_error"],
        s=14,
        zorder=5,
    )
    axes[1].set_ylabel("MAE")
    axes[1].set_xlabel("Timestamp")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    safe_savefig(fig, AE_GRAPH, dpi=150, bbox_inches="tight")
    plt.close(fig)



def predict_dataframe(
    model,
    F_target: pd.DataFrame,
    F_history: pd.DataFrame,
    seq_data: dict,
    split_name: str = "test",
) -> pd.DataFrame:
    """Score a complete split household-by-household without a global RAM spike."""
    preprocessing = {
        "feature_cols": seq_data["feature_cols"],
        "train_medians": seq_data["train_medians"],
        "scaler": seq_data["scaler"],
    }
    parts = []
    for hh_id in sorted(F_target[COL_HOUSEHOLD].unique()):
        target_hh = F_target[F_target[COL_HOUSEHOLD] == hh_id].copy().sort_index()
        history_hh = F_history[F_history[COL_HOUSEHOLD] == hh_id].copy().sort_index()
        X, meta = build_sequences_for_split(
            target_hh,
            history_hh,
            preprocessing,
            max_sequences=None,
            seed=0,
        )
        if len(X) == 0:
            continue
        part = predict_on_split(
            model, X, meta, seq_data, split_name=split_name
        )
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(
        [COL_HOUSEHOLD, "timestamp"]
    ).reset_index(drop=True)


def verify_us12(results: pd.DataFrame) -> bool:
    checks = [
        ("AE results not empty", not results.empty),
        ("reconstruction error", "recon_error" in results.columns),
        ("anomaly score", "anomaly_score" in results.columns),
        ("is_anomaly", "is_anomaly" in results.columns),
        ("model saved", os.path.exists(AE_MODEL_PATH)),
        ("threshold saved", os.path.exists(AE_THRESHOLD_PKL)),
    ]
    ok = all(value for _, value in checks)
    for name, passed in checks:
        log_print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    log_print(f"US-12 status: {'PASS' if ok else 'FAIL'}")
    return bool(ok)

def run_ae_pipeline(splits: dict) -> dict:
    """
    Standalone US-12 pipeline using a train-percentile fallback threshold.

    The full runner later replaces this threshold with a validation-calibrated
    threshold before final test/API output.
    """
    seq_data = prepare_split_sequences(splits["train"], splits["val"], splits["test"])
    model = train_autoencoder(seq_data)
    threshold = compute_error_reference(model, seq_data)

    results_val = predict_dataframe(
        model,
        splits["val"],
        splits["train"],
        seq_data,
        "val",
    )
    test_history = pd.concat([splits["train"], splits["val"]], axis=0)
    results_test = predict_dataframe(
        model,
        splits["test"],
        test_history,
        seq_data,
        "test",
    )
    results_test.to_csv(AE_RESULTS_CSV, index=False)
    log_print(
        f"Test AE anomalies: {int(results_test['is_anomaly'].sum()):,} / "
        f"{len(results_test):,}"
    )
    log_print(f"Saved: {AE_RESULTS_CSV}")
    plot_ae_evidence(results_test)
    verify_us12(results_test)

    return {
        "model": model,
        "threshold": threshold,
        "scaler": seq_data["scaler"],
        "feature_cols": seq_data["feature_cols"],
        "results_val": results_val,
        "results_test": results_test,
        "sequence_data": seq_data,
    }
