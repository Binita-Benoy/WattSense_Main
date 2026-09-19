from logging_utils import log_print
import logging
logger = logging.getLogger()
# US-13: TEST-split injection evaluation for both anomaly detectors.
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from config import (
    EVAL_CSV, EVAL_GRAPH, COL_HOUSEHOLD, COL_KWH,
    IF_MODEL_PKL, IF_SCALER_PKL, AE_SCALER_PKL, AE_THRESHOLD_PKL
)


def _compute_metrics(y_true, y_pred, scores, name, hh_id):
    common = y_true.index.intersection(y_pred.index)
    y_t, y_p, s = y_true.loc[common], y_pred.loc[common], scores.loc[common]
    if y_t.sum() == 0:
        return {"detector": name, COL_HOUSEHOLD: hh_id,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "auc_roc": 0.5, "support": 0}
    prec = precision_score(y_t, y_p, zero_division=0)
    rec = recall_score(y_t, y_p, zero_division=0)
    f1 = f1_score(y_t, y_p, zero_division=0)
    try:
        auc = roc_auc_score(y_t, s)
    except Exception:
        auc = 0.5
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"detector": name, COL_HOUSEHOLD: hh_id,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "auc_roc": round(auc, 4),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "support": int(y_t.sum())}


def _refresh_injected_features(df):
    """Refresh energy-derived features after synthetic kWh injection."""
    from feature_engineering.us10_features import (
        add_lag_features, add_rolling_statistics, add_peak_indicator,
        add_deviation_features
    )
    out = df.copy().sort_index()
    out = add_lag_features(out)
    out = add_rolling_statistics(out)
    out = add_peak_indicator(out)
    out = add_deviation_features(out)
    numeric = out.select_dtypes(include=[np.number]).columns
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    out[numeric] = out[numeric].fillna(out[numeric].median())
    return out


def evaluate_on_injection(splits, if_trained=None, ae_model=None, ae_threshold=None):
    log_print("\n" + "=" * 70)
    log_print("US-13: INJECTION-BASED EVALUATION ON TEST SPLIT")
    log_print("=" * 70)

    from anomaly_detection.us11_isolation_forest import IF_FEATURES
    from anomaly_detection.us12_autoencoder import AE_FEATURES, make_sequences

    model_if = joblib.load(IF_MODEL_PKL)
    scaler_if = joblib.load(IF_SCALER_PKL)
    scaler_ae = joblib.load(AE_SCALER_PKL)
    threshold = joblib.load(AE_THRESHOLD_PKL) if ae_threshold is None else ae_threshold

    all_metrics = []
    for hh_id in sorted(splits["test"][COL_HOUSEHOLD].unique()):
        df_hh = splits["test"][splits["test"][COL_HOUSEHOLD] == hh_id].copy().sort_index()
        if len(df_hh) < 300:
            continue

        rng = np.random.default_rng(42 + hh_id % 100)
        n = len(df_hh)
        labels = np.zeros(n, dtype=int)
        safe = np.arange(200, n - 10)
        n_spikes = max(5, int(n * 0.01))
        n_dropouts = max(3, int(n * 0.005))
        spike_idx = rng.choice(safe, size=min(n_spikes, len(safe)), replace=False)
        dropout_idx = rng.choice(np.setdiff1d(safe, spike_idx), size=min(n_dropouts, max(1, len(safe)-len(spike_idx))), replace=False)
        for idx in spike_idx:
            df_hh.iloc[idx, df_hh.columns.get_loc(COL_KWH)] *= rng.uniform(5, 8)
            labels[idx] = 1
        for idx in dropout_idx:
            df_hh.iloc[idx, df_hh.columns.get_loc(COL_KWH)] *= 0.02
            labels[idx] = 1

        df_injected = _refresh_injected_features(df_hh)
        y_true = pd.Series(labels, index=df_hh.index)

        # Isolation Forest: same features and scaler learned from TRAIN only.
        cols_if = [c for c in IF_FEATURES if c in df_injected.columns]
        X_if = df_injected[cols_if].astype(float).replace([np.inf, -np.inf], np.nan)
        X_if = X_if.fillna(df_injected[cols_if].median())
        X_if_scaled = scaler_if.transform(X_if)
        raw = model_if.predict(X_if_scaled)
        raw_scores = model_if.score_samples(X_if_scaled)
        scores = 1 - ((raw_scores - raw_scores.min()) /
                      (raw_scores.max() - raw_scores.min() + 1e-9))
        y_pred = pd.Series((raw == -1).astype(int), index=df_hh.index)
        score_series = pd.Series(scores, index=df_hh.index)
        all_metrics.append(_compute_metrics(y_true, y_pred, score_series, "Isolation Forest", hh_id))

        # Multivariate LSTM Autoencoder.
        if ae_model is not None:
            cols_ae = [c for c in AE_FEATURES if c in df_injected.columns]
            X_vals = df_injected[cols_ae].astype(float).replace([np.inf, -np.inf], np.nan)
            X_vals = X_vals.fillna(X_vals.median())
            scaled = scaler_ae.transform(X_vals)
            X_ae, idx_list = make_sequences(scaled)
            if len(X_ae):
                pred = ae_model.predict(X_ae, batch_size=256, verbose=0)
                errors = np.mean(np.abs(X_ae - pred), axis=(1, 2))
                norm = (errors - errors.min()) / (errors.max() - errors.min() + 1e-9)
                ae_flags = (errors > threshold).astype(int)
                ts = df_hh.index
                ae_index = [ts[j] for j in idx_list]
                yp = pd.Series(ae_flags, index=ae_index)
                sc = pd.Series(norm, index=ae_index)
                yt = y_true.reindex(ae_index).fillna(0)
                all_metrics.append(_compute_metrics(yt, yp, sc, "LSTM Autoencoder", hh_id))

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(EVAL_CSV, index=False)
    _print_aggregate(metrics)
    plot_evaluation_chart(metrics)
    return metrics


def _print_aggregate(df):
    log_print("\nAGGREGATE TEST METRICS")
    for det in ["Isolation Forest", "LSTM Autoencoder"]:
        sub = df[df["detector"] == det]
        if sub.empty:
            continue
        log_print(det)
        for m in ["precision", "recall", "f1", "auc_roc"]:
            log_print(f"  {m}: {sub[m].mean():.4f} ± {sub[m].std():.4f}")


def plot_evaluation_chart(df):
    if df.empty:
        return
    metrics = ["precision", "recall", "f1"]
    detectors = list(df["detector"].unique())
    hh_ids = sorted(df[COL_HOUSEHOLD].unique())
    x = np.arange(len(hh_ids))
    w = 0.8 / max(1, len(detectors))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, metrics):
        for i, det in enumerate(detectors):
            sub = df[df["detector"] == det]
            vals = [sub.loc[sub[COL_HOUSEHOLD] == h, metric].iloc[0]
                    if not sub.loc[sub[COL_HOUSEHOLD] == h].empty else 0 for h in hh_ids]
            ax.bar(x + (i - (len(detectors)-1)/2) * w, vals, w, label=det)
        ax.set_title(metric.upper())
        ax.set_xticks(x)
        ax.set_xticklabels([f"HH{i+1}" for i in range(len(hh_ids))])
        ax.set_ylim(0, 1.15)
        ax.grid(alpha=0.2, axis="y")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(EVAL_GRAPH, dpi=150, bbox_inches="tight")
    plt.close()


def verify_us13(df_metrics):
    checks = [
        "Isolation Forest" in df_metrics.get("detector", pd.Series()).values,
        "LSTM Autoencoder" in df_metrics.get("detector", pd.Series()).values,
        os.path.exists(EVAL_CSV)
    ]
    ok = all(checks)
    log_print(f"US-13 status: {'PASS' if ok else 'FAIL'}")
    return ok
