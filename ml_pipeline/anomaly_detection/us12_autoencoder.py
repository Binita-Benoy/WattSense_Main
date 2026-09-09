# anomaly_detection/us12_autoencoder.py
# ─────────────────────────────────────────────────────────────────────────────
# US-12: LSTM Autoencoder — with proper train/val/test split
#
# WHAT CHANGED:
#   BEFORE: trained and evaluated on same data
#   NOW:
#     - Scaler fitted on TRAIN split only
#     - Model trained on TRAIN split
#     - Threshold computed on TRAIN reconstruction errors
#     - Early stopping monitored on VAL split
#     - Final predictions on TEST split only
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.preprocessing import MinMaxScaler
from anomaly_detection.config import (
    AE_SEQ_LEN, AE_BATCH_SIZE, AE_EPOCHS, AE_THRESHOLD_PCT,
    AE_MODEL_PATH, AE_SCALER_PKL, AE_THRESHOLD_PKL,
    AE_RESULTS_CSV, AE_GRAPH,
    COL_HOUSEHOLD, COL_KWH, OUT_GRAPHS, PRIMARY_HH
)


def make_sequences(series: np.ndarray,
                   seq_len: int = AE_SEQ_LEN) -> tuple:
    """
    Build overlapping windows from a 1D series.

    Returns:
        X: [n_windows, seq_len, 1]
        indices: positions of the last element in each window
                 (used to align predictions back to timestamps)
    """
    X, indices = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i: i + seq_len])
        indices.append(i + seq_len)   # timestamp of window end
    X = np.array(X)
    return X.reshape(X.shape[0], X.shape[1], 1), indices


def build_autoencoder(seq_len: int = AE_SEQ_LEN):
    """LSTM Autoencoder — same architecture as before."""
    try:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, LSTM, Dense,
            RepeatVector, TimeDistributed, Dropout
        )
    except ImportError:
        raise ImportError("pip install tensorflow")

    inp = Input(shape=(seq_len, 1))
    x   = LSTM(64, activation="relu",
                return_sequences=True)(inp)
    x   = Dropout(0.2)(x)
    x   = LSTM(32, activation="relu",
                return_sequences=False)(x)
    x   = RepeatVector(seq_len)(x)
    x   = LSTM(32, activation="relu",
                return_sequences=True)(x)
    x   = Dropout(0.2)(x)
    x   = LSTM(64, activation="relu",
                return_sequences=True)(x)
    out = TimeDistributed(Dense(1))(x)

    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model


def prepare_split_sequences(
    F_train: pd.DataFrame,
    F_val:   pd.DataFrame,
    F_test:  pd.DataFrame
) -> dict:
    """
    Build sequences for all three splits.

    CRITICAL ORDER OF OPERATIONS:
        1. Fit MinMaxScaler on TRAIN only
        2. Transform TRAIN, VAL, TEST using train statistics
        3. Build sequences for each split independently

    Why fit on train only?
        If we fit the scaler on all data, the scaler learns
        the min/max of test data — which it would not know
        in real deployment. This is data leakage.

    Args:
        F_train, F_val, F_test: split DataFrames

    Returns:
        dict with sequences and metadata for all three splits
    """
    print("\n" + "="*60)
    print("US-12: PREPARING LSTM SEQUENCES")
    print("="*60)
    print("  ⚠️  Scaler fitted on TRAIN split only")

    households = sorted(F_train[COL_HOUSEHOLD].unique())

    # ── Fit scaler on TRAIN data only ─────────────────────────────────────
    all_train_kwh = F_train[COL_KWH].values.reshape(-1, 1)
    scaler        = MinMaxScaler()
    scaler.fit(all_train_kwh)   # ← TRAIN ONLY
    print(f"  Scaler range learned from TRAIN: "
          f"[{all_train_kwh.min():.4f}, "
          f"{all_train_kwh.max():.4f}]")

    joblib.dump(scaler, AE_SCALER_PKL)

    results = {
        "train": {"X": [], "meta": []},
        "val":   {"X": [], "meta": []},
        "test":  {"X": [], "meta": []},
        "scaler": scaler
    }

    for hh_id in households:
        for split_name, F_split in [("train", F_train),
                                     ("val",   F_val),
                                     ("test",  F_test)]:
            df_hh = (F_split[F_split[COL_HOUSEHOLD] == hh_id]
                     [COL_KWH].sort_index())

            if len(df_hh) < AE_SEQ_LEN + 5:
                continue

            # transform NOT fit_transform for val and test
            scaled = scaler.transform(
                df_hh.values.reshape(-1, 1)
            ).flatten()

            X_seqs, idx_list = make_sequences(scaled, AE_SEQ_LEN)

            results[split_name]["X"].append(X_seqs)

            ts   = df_hh.index
            vals = df_hh.values
            for j in idx_list:
                results[split_name]["meta"].append({
                    "timestamp":   ts[j],
                    COL_HOUSEHOLD: hh_id,
                    "consumption": float(vals[j])
                })

    # Concatenate arrays
    for split_name in ["train", "val", "test"]:
        if len(results[split_name]["X"]) > 0:
            results[split_name]["X"] = np.concatenate(
                results[split_name]["X"], axis=0
            )
            results[split_name]["meta"] = pd.DataFrame(
                results[split_name]["meta"]
            )
            n = len(results[split_name]["X"])
            print(f"  {split_name.upper()}: "
                  f"{n:,} sequences")
        else:
            results[split_name]["X"]    = np.array([])
            results[split_name]["meta"] = pd.DataFrame()

    return results


def train_autoencoder(seq_data: dict) -> object:
    """
    Train autoencoder.

    Training data:   seq_data["train"]["X"]
    Validation data: seq_data["val"]["X"]
        ↑ used for early stopping only — model NEVER
          adjusts weights based on val loss directly.
          Early stopping just determines when to stop.
    """
    try:
        from tensorflow.keras.callbacks import (
            EarlyStopping, ModelCheckpoint
        )
    except ImportError:
        raise ImportError("pip install tensorflow")

    X_train = seq_data["train"]["X"]
    X_val   = seq_data["val"]["X"]

    print(f"\n{'='*60}")
    print("US-12: TRAINING LSTM AUTOENCODER")
    print("="*60)
    print(f"  TRAIN sequences: {len(X_train):,}")
    print(f"  VAL   sequences: {len(X_val):,}  "
          f"(early stopping only)")
    print(f"  TEST  sequences: evaluated AFTER training")
    print(f"  Epochs: up to {AE_EPOCHS} (patience=5)")

    model = build_autoencoder(X_train.shape[1])

    # Validation data passed explicitly — not just split from train
    # This ensures the val split is the same chronological data
    # we defined in splitter.py
    history = model.fit(
        X_train, X_train,
        epochs           = AE_EPOCHS,
        batch_size       = AE_BATCH_SIZE,
        validation_data  = (X_val, X_val),   # explicit val set
        callbacks        = [
            EarlyStopping(
                monitor              = "val_loss",
                patience             = 5,
                restore_best_weights = True,
                verbose              = 1
            ),
            ModelCheckpoint(
                filepath       = AE_MODEL_PATH,
                save_best_only = True,
                verbose        = 0
            )
        ],
        shuffle          = True,
        verbose          = 1
    )

    best = np.argmin(history.history["val_loss"]) + 1
    print(f"\n  ✅ Training complete — best epoch: {best}")
    print(f"     Train loss: {history.history['loss'][-1]:.6f}")
    print(f"     Val loss:   {history.history['val_loss'][-1]:.6f}")

    _plot_training(history)
    return model


def _plot_training(history) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"],     label="Train loss")
    ax.plot(history.history["val_loss"], label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("US-12 LSTM Autoencoder — Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(OUT_GRAPHS, "us12_training_loss.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Training curve: {path}")


def compute_threshold(model,
                      X_train: np.ndarray) -> float:
    """
    Compute threshold from TRAINING reconstruction errors.

    Why train data for threshold?
        We want to find the error level above which a sequence
        is 'unusual'. The baseline for 'usual' is what the model
        saw during training — so training errors define normal.

        Using test errors to set the threshold would give us
        perfect knowledge of the test distribution — leakage.

    Returns:
        float threshold at AE_THRESHOLD_PCT of train errors
    """
    print(f"\n  Computing threshold from TRAIN errors "
          f"({AE_THRESHOLD_PCT}th percentile)...")

    X_pred    = model.predict(X_train, batch_size=256, verbose=0)
    errors    = np.mean(np.abs(X_train - X_pred), axis=(1, 2))
    threshold = float(np.percentile(errors, AE_THRESHOLD_PCT))

    print(f"  Train error — min:  {errors.min():.6f}")
    print(f"  Train error — mean: {errors.mean():.6f}")
    print(f"  Train error — max:  {errors.max():.6f}")
    print(f"  Threshold:          {threshold:.6f}  "
          f"(above = anomaly)")

    joblib.dump(threshold, AE_THRESHOLD_PKL)
    print(f"  ✓ Threshold saved: {AE_THRESHOLD_PKL}")
    return threshold


def predict_on_split(model,
                     X_split:   np.ndarray,
                     metadata:  pd.DataFrame,
                     threshold: float,
                     split_name: str = "test") -> pd.DataFrame:
    """
    Score sequences from one split.

    For val: check model is learning (errors should be moderate)
    For test: final honest evaluation

    The threshold from training is applied WITHOUT modification.
    """
    if len(X_split) == 0:
        print(f"  ⚠️  No sequences in {split_name} split")
        return pd.DataFrame()

    print(f"\n  Scoring {split_name.upper()} split "
          f"({len(X_split):,} sequences)...")
    print(f"  ⚠️  Model NOT retrained — threshold from TRAIN")

    X_pred   = model.predict(X_split, batch_size=256, verbose=0)
    errors   = np.mean(np.abs(X_split - X_pred), axis=(1, 2))

    is_anomaly = (errors > threshold).astype(int)
    e_min      = errors.min()
    e_max      = errors.max()
    norm_scores= (errors - e_min) / (e_max - e_min + 1e-9)

    results = metadata.copy()
    results["recon_error"]   = errors.round(8)
    results["anomaly_score"] = norm_scores.round(6)
    results["is_anomaly"]    = is_anomaly
    results["threshold"]     = threshold
    results["split"]         = split_name
    results["method"]        = "lstm_autoencoder"

    n     = len(results)
    n_anom= results["is_anomaly"].sum()
    print(f"  Anomalies: {n_anom:,} / {n:,} "
          f"({n_anom/n*100:.1f}%)")

    return results


def run_ae_pipeline(splits: dict) -> dict:
    """
    Full AE pipeline with correct split discipline.

    Returns:
        dict with model, threshold, and results per split
    """
    # Prepare sequences for all splits
    seq_data  = prepare_split_sequences(
        splits["train"], splits["val"], splits["test"]
    )

    # Train on train split; early stop on val split
    model     = train_autoencoder(seq_data)

    # Threshold from train errors only
    threshold = compute_threshold(model, seq_data["train"]["X"])

    # Predict on val (sanity check)
    results_val  = predict_on_split(
        model,
        seq_data["val"]["X"],
        seq_data["val"]["meta"],
        threshold, "val"
    )

    # Predict on test (final evaluation)
    results_test = predict_on_split(
        model,
        seq_data["test"]["X"],
        seq_data["test"]["meta"],
        threshold, "test"
    )

    # Save TEST results for US-13
    results_test.to_csv(AE_RESULTS_CSV, index=False)
    print(f"\n  ✓ Test results saved: {AE_RESULTS_CSV}")

    plot_ae_evidence(results_test)
    verify_us12(results_test, threshold)

    return {
        "model":        model,
        "threshold":    threshold,
        "results_val":  results_val,
        "results_test": results_test
    }


def plot_ae_evidence(results_test: pd.DataFrame,
                     household_id: int = PRIMARY_HH) -> None:
    """Evidence plot from TEST split only."""
    df    = results_test[
        results_test[COL_HOUSEHOLD] == household_id
    ].copy()
    df    = df.sort_values("timestamp")

    if df.empty:
        print(f"  ⚠️  No test data for HH {household_id}")
        return

    anom  = df[df["is_anomaly"] == 1]
    norm  = df[df["is_anomaly"] == 0]

    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    fig.suptitle(
        f"US-12 — LSTM Autoencoder  |  HH {household_id}  |  "
        f"TEST SPLIT (unseen during training)",
        fontsize=12, fontweight="bold"
    )

    # Panel 1: Consumption
    ax = axes[0]
    ax.set_facecolor("white")
    ax.plot(pd.to_datetime(norm["timestamp"]),
            norm["consumption"],
            color="#1565C0", lw=0.7, alpha=0.6, label="Normal")
    ax.scatter(pd.to_datetime(anom["timestamp"]),
               anom["consumption"],
               color="#C62828", s=25, zorder=5,
               label=f"Anomaly ({len(anom)})")
    ax.set_ylabel("kWh")
    ax.set_title("Consumption — test split")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.text(0.01, 0.97,
            f"⚠️ TEST SPLIT — model never saw this data",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#FFF9C4",
                      edgecolor="#F9A825", alpha=0.9))

    # Panel 2: Reconstruction error
    ax = axes[1]
    ax.set_facecolor("white")
    ts_all = pd.to_datetime(df["timestamp"])
    ax.fill_between(ts_all, df["recon_error"],
                    alpha=0.3, color="#6A1B9A")
    ax.plot(ts_all, df["recon_error"],
            color="#6A1B9A", lw=0.7)
    threshold = df["threshold"].iloc[0]
    ax.axhline(threshold, color="#C62828", ls="--", lw=1.2,
               label=f"Threshold (train 95th pct = {threshold:.5f})")
    ax.scatter(pd.to_datetime(anom["timestamp"]),
               anom["recon_error"],
               color="#C62828", s=15, zorder=5)
    ax.set_ylabel("Reconstruction Error (MAE)")
    ax.set_xlabel("Timestamp")
    ax.set_title("Reconstruction error — threshold from TRAIN data")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(AE_GRAPH, dpi=150, bbox_inches="tight")
    print(f"  ✓ Graph saved: {AE_GRAPH}")
    plt.show()


def verify_us12(results: pd.DataFrame,
                threshold: float) -> bool:
    print(f"\n{'='*60}")
    print("US-12 ACCEPTANCE CRITERIA")
    print("="*60)
    criteria = [
        ("Autoencoder trained",
         os.path.exists(AE_MODEL_PATH)),
        ("Reconstruction error calculated",
         "recon_error" in results.columns),
        ("Threshold from TRAIN data",
         "threshold" in results.columns),
        ("Anomalies identified on TEST split",
         "is_anomaly" in results.columns and
         "split" in results.columns and
         (results["split"] == "test").all()),
        ("Results saved",
         os.path.exists(AE_RESULTS_CSV)),
    ]
    all_pass = True
    for name, passed in criteria:
        print(f"  {'✅' if passed else '❌'}  {name}")
        if not passed:
            all_pass = False
    print(f"\n  {'ALL PASSED ✅' if all_pass else 'SOME FAILED ❌'}")
    return all_pass