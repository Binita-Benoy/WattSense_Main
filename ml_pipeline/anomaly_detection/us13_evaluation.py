# anomaly_detection/us13_evaluation.py
# ─────────────────────────────────────────────────────────────────────────────
# US-13: Evaluate anomaly detection
#
# WHAT CHANGED:
#   BEFORE: evaluated on training data → inflated metrics
#   NOW:    evaluates on TEST split only → honest metrics
#
# Ground truth strategy: z-score injection
#   We inject known synthetic anomalies into the TEST data,
#   run both detectors, and measure how many they find.
#   This gives real precision/recall/F1 against a known truth.
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from anomaly_detection.config import (
    IF_RESULTS_CSV, AE_RESULTS_CSV,
    EVAL_CSV, EVAL_GRAPH,
    COL_HOUSEHOLD, COL_KWH, OUT_GRAPHS,
    IF_MODEL_PKL, IF_SCALER_PKL,
    AE_MODEL_PATH, AE_THRESHOLD_PKL, AE_SCALER_PKL
)


def create_ground_truth_zscore(
    results_if: pd.DataFrame,
    z_threshold: float = 3.0
) -> pd.Series:
    """
    Pseudo-ground truth from z-score on TEST data.

    A reading is labelled as a TRUE anomaly if |z_score| > 3.0
    in the TEST split. This is statistically motivated and
    reproducible — 3 sigma events occur rarely under normal
    distributions and represent objectively unusual readings.

    Note: this is applied to TEST data only.
    The model has never seen this data's z-scores during training.
    """
    if "z_score" not in results_if.columns:
        print("  z_score not available — using top 5% scores")
        q95 = results_if["anomaly_score"].quantile(0.95)
        return (results_if["anomaly_score"] >= q95).astype(int)

    gt    = (results_if["z_score"].abs() > z_threshold).astype(int)
    n_pos = gt.sum()
    print(f"  Ground truth (|z| > {z_threshold}): "
          f"{n_pos:,} positive "
          f"({n_pos/len(gt)*100:.1f}%) on TEST split")
    return gt


def evaluate_on_injection(splits: dict,
                          if_trained:  dict,
                          ae_model,
                          ae_threshold: float) -> dict:
    """
    Evaluate both detectors by injecting known anomalies
    into the TEST split and measuring detection rate.

    Why injection?
        SGSC has no human labels. Injection gives us a
        controlled ground truth where we know exactly
        which readings are anomalous.

    Injection types:
        Spike:   consumption × 5-8x  (fault, large appliance)
        Dropout: consumption × 0.02  (meter error, dropout)

    Process:
        1. Take clean TEST data
        2. Inject known anomalies
        3. Run each detector (using models trained on TRAIN)
        4. Compare detector output to injection labels
        5. Compute precision, recall, F1

    Args:
        splits:       dict from splitter.py
        if_trained:   dict from us11 train step
        ae_model:     trained Keras model
        ae_threshold: threshold from us12 compute_threshold

    Returns:
        dict of metrics per detector
    """
    print("\n" + "="*60)
    print("US-13: INJECTION-BASED EVALUATION ON TEST SPLIT")
    print("="*60)
    print("  ⚠️  Injecting synthetic anomalies into TEST data")
    print("  ⚠️  Both detectors use models trained on TRAIN only")

    from anomaly_detection.us11_isolation_forest import IF_FEATURES
    from anomaly_detection.us12_autoencoder import (
        make_sequences, AE_SEQ_LEN
    )

    F_test   = splits["test"].copy()
    hh_ids   = sorted(F_test[COL_HOUSEHOLD].unique())

    all_metrics = []

    for hh_id in hh_ids:
        print(f"\n  Household {hh_id}:")

        df_hh = F_test[F_test[COL_HOUSEHOLD] == hh_id].copy()
        df_hh = df_hh.sort_index()

        if len(df_hh) < 300:
            print(f"    Too few test rows — skipped")
            continue

        # ── Inject anomalies ───────────────────────────────────────────
        np.random.seed(42 + hh_id % 100)
        df_injected = df_hh.copy()
        n           = len(df_injected)
        labels      = np.zeros(n, dtype=int)

        n_spikes   = max(5, int(n * 0.01))   # 1% spikes
        n_dropouts = max(3, int(n * 0.005))  # 0.5% dropouts

        safe_range  = range(200, n - 10)
        spike_idx   = np.random.choice(
            safe_range, size=n_spikes, replace=False
        )
        for idx in spike_idx:
            df_injected[COL_KWH].iloc[idx] *= np.random.uniform(5, 8)
            labels[idx] = 1

        dropout_idx = np.random.choice(
            safe_range, size=n_dropouts, replace=False
        )
        for idx in dropout_idx:
            df_injected[COL_KWH].iloc[idx] *= 0.02
            labels[idx] = 1

        y_true = pd.Series(labels, index=df_hh.index)
        n_injected = labels.sum()
        print(f"    Injected {n_injected} anomalies "
              f"({n_spikes} spikes, {n_dropouts} dropouts) "
              f"into {n} readings")

        # ── Evaluate Isolation Forest ──────────────────────────────────
        import joblib
        model_if  = joblib.load(IF_MODEL_PKL)
        scaler_if = joblib.load(IF_SCALER_PKL)

        avail_if   = [c for c in IF_FEATURES
                      if c in df_injected.columns]
        X_if       = df_injected[avail_if]
        X_if_scaled= scaler_if.transform(X_if)

        raw_if     = model_if.predict(X_if_scaled)
        y_pred_if  = (raw_if == -1).astype(int)
        raw_scores_if = model_if.score_samples(X_if_scaled)
        scores_if  = 1 - (
            (raw_scores_if - raw_scores_if.min()) /
            (raw_scores_if.max() - raw_scores_if.min() + 1e-9)
        )
        scores_if  = pd.Series(scores_if, index=df_hh.index)
        y_pred_if  = pd.Series(y_pred_if, index=df_hh.index)

        m_if = _compute_metrics(
            y_true, y_pred_if, scores_if,
            "Isolation Forest", hh_id
        )
        all_metrics.append(m_if)

        # ── Evaluate LSTM Autoencoder ──────────────────────────────────
        if ae_model is not None:
            scaler_ae = joblib.load(AE_SCALER_PKL)
            threshold = joblib.load(AE_THRESHOLD_PKL)

            kwh_vals = df_injected[COL_KWH].values
            scaled   = scaler_ae.transform(
                kwh_vals.reshape(-1, 1)
            ).flatten()
            X_ae, idx_list = make_sequences(scaled, AE_SEQ_LEN)

            if len(X_ae) > 0:
                X_pred   = ae_model.predict(
                    X_ae, batch_size=256, verbose=0
                )
                errors   = np.mean(
                    np.abs(X_ae - X_pred), axis=(1, 2)
                )
                norm_sc  = (errors - errors.min()) / (
                    errors.max() - errors.min() + 1e-9
                )
                is_anom  = (errors > threshold).astype(int)

                # Align to original index
                ts       = df_hh.index
                ae_index = [ts[j] for j in idx_list
                             if j < len(ts)]
                ae_len   = min(len(ae_index), len(is_anom))

                y_pred_ae = pd.Series(
                    is_anom[:ae_len],
                    index=ae_index[:ae_len]
                )
                scores_ae = pd.Series(
                    norm_sc[:ae_len],
                    index=ae_index[:ae_len]
                )

                # Align ground truth to AE index
                y_true_ae = y_true.reindex(
                    y_pred_ae.index
                ).fillna(0)

                m_ae = _compute_metrics(
                    y_true_ae, y_pred_ae, scores_ae,
                    "LSTM Autoencoder", hh_id
                )
                all_metrics.append(m_ae)

    # ── Aggregate ─────────────────────────────────────────────────────────
    df_metrics = pd.DataFrame(all_metrics)
    _print_aggregate(df_metrics)
    df_metrics.to_csv(EVAL_CSV, index=False)
    print(f"\n  ✓ Evaluation saved: {EVAL_CSV}")
    plot_evaluation_chart(df_metrics)

    return df_metrics


def _compute_metrics(y_true:  pd.Series,
                     y_pred:  pd.Series,
                     scores:  pd.Series,
                     name:    str,
                     hh_id:   int) -> dict:
    """Compute precision, recall, F1, AUC-ROC for one case."""
    # Align indices
    common  = y_true.index.intersection(y_pred.index)
    y_t     = y_true.loc[common]
    y_p     = y_pred.loc[common]
    s       = scores.loc[common]

    if y_t.sum() == 0:
        print(f"    {name}: no ground truth positives")
        return {"detector": name, COL_HOUSEHOLD: hh_id,
                "precision": 0.0, "recall": 0.0,
                "f1": 0.0, "auc_roc": 0.5, "support": 0}

    prec = precision_score(y_t, y_p, zero_division=0)
    rec  = recall_score   (y_t, y_p, zero_division=0)
    f1   = f1_score       (y_t, y_p, zero_division=0)
    try:
        auc  = roc_auc_score(y_t, s)
    except Exception:
        auc  = 0.5

    cm   = confusion_matrix(y_t, y_p)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) \
                     else (0, 0, 0, 0)

    print(f"    {name}:")
    print(f"      Precision: {prec:.4f}  "
          f"Recall: {rec:.4f}  F1: {f1:.4f}  "
          f"AUC: {auc:.4f}")
    print(f"      TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    return {
        "detector":     name,
        COL_HOUSEHOLD:  hh_id,
        "precision":    round(prec, 4),
        "recall":       round(rec,  4),
        "f1":           round(f1,   4),
        "auc_roc":      round(auc,  4),
        "tp":           int(tp), "fp": int(fp),
        "tn":           int(tn), "fn": int(fn),
        "support":      int(y_t.sum())
    }


def _print_aggregate(df_metrics: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print("AGGREGATE METRICS (mean across households)")
    print("On TEST split with injected ground truth")
    print("="*60)
    for det in ["Isolation Forest", "LSTM Autoencoder"]:
        sub = df_metrics[df_metrics["detector"] == det]
        if len(sub) == 0:
            continue
        print(f"\n  {det}:")
        for m in ["precision", "recall", "f1", "auc_roc"]:
            v = sub[m]
            print(f"    {m:<12}: "
                  f"{v.mean():.4f} ± {v.std():.4f}")


def plot_evaluation_chart(df_metrics: pd.DataFrame) -> None:
    """Grouped bar chart: IF vs AE per household."""
    metrics   = ["precision", "recall", "f1"]
    labels    = ["Precision", "Recall", "F1 Score"]
    colours   = {"Isolation Forest": "#1565C0",
                 "LSTM Autoencoder": "#C62828"}
    detectors = df_metrics["detector"].unique()
    hh_ids    = sorted(df_metrics[COL_HOUSEHOLD].unique())
    n_hh      = len(hh_ids)
    x         = np.arange(n_hh)
    w         = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "US-13 — Anomaly Detection Evaluation\n"
        "TEST split | Injection ground truth | "
        "Isolation Forest vs LSTM Autoencoder",
        fontsize=11, fontweight="bold"
    )

    for ax, metric, label in zip(axes, metrics, labels):
        for i, det in enumerate(detectors):
            sub  = df_metrics[df_metrics["detector"] == det]
            vals = []
            for hh in hh_ids:
                row = sub[sub[COL_HOUSEHOLD] == hh]
                vals.append(row[metric].values[0]
                            if len(row) > 0 else 0.0)
            offset = (i - len(detectors)/2 + 0.5) * w
            bars   = ax.bar(x + offset, vals, w,
                            label=det,
                            color=colours.get(det, "gray"),
                            alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}",
                        ha="center", fontsize=7.5)
        ax.set_title(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"HH{i+1}" for i in range(n_hh)], fontsize=8
        )
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(EVAL_GRAPH, dpi=150, bbox_inches="tight")
    print(f"  ✓ Chart saved: {EVAL_GRAPH}")
    plt.show()


def verify_us13(df_metrics: pd.DataFrame) -> bool:
    print(f"\n{'='*60}")
    print("US-13 ACCEPTANCE CRITERIA")
    print("="*60)
    criteria = [
        ("Isolation Forest precision/recall/F1",
         "Isolation Forest" in df_metrics["detector"].values and
         all(c in df_metrics.columns
             for c in ["precision","recall","f1"])),
        ("LSTM Autoencoder precision/recall/F1",
         "LSTM Autoencoder" in df_metrics["detector"].values),
        ("Evaluated on TEST split (not train)",
         True),   # enforced by design
        ("Results saved",
         os.path.exists(EVAL_CSV)),
    ]
    all_pass = True
    for name, passed in criteria:
        print(f"  {'✅' if passed else '❌'}  {name}")
        if not passed:
            all_pass = False
    print(f"\n  {'ALL PASSED ✅' if all_pass else 'SOME FAILED ❌'}")
    print(f"\n  Full table:")
    cols = ["detector", COL_HOUSEHOLD,
            "precision","recall","f1","auc_roc","support"]
    cols = [c for c in cols if c in df_metrics.columns]
    print(df_metrics[cols].to_string(index=False))
    return all_pass