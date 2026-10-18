
from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import (
    CALIBRATION_CSV,
    CALIBRATION_RANDOM_STATE,
    COL_HOUSEHOLD,
    COL_KWH,
    ENSEMBLE_BETA,
    ENSEMBLE_CALIBRATION_REPEATS,
    ENSEMBLE_CONFIG_PKL,
    EVAL_BY_TYPE_CSV,
    EVAL_CSV,
    EVAL_GRAPH,
    EVAL_SUMMARY_CSV,
    INJECTION_DROPOUT_FRACTION,
    INJECTION_DROPOUT_MULT,
    INJECTION_MIN_DROPOUTS,
    INJECTION_MIN_SPIKES,
    INJECTION_SPIKE_FRACTION,
    INJECTION_SPIKE_MAX_MULT,
    INJECTION_SPIKE_MIN_MULT,
    TEST_INJECTION_RANDOM_STATE,
)
from logging_utils import log_print, safe_savefig
from anomaly_detection.ensemble_detector import (
    combine_scores,
    load_ensemble_config,
    save_ensemble_config,
    search_ensemble_parameters,
)

logger = logging.getLogger()


def _safe_auc(metric_fn, y_true, scores, default=0.5):
    try:
        if len(np.unique(y_true)) < 2:
            return default
        return float(metric_fn(y_true, scores))
    except Exception:
        return default


def _compute_metrics(y_true, y_pred, scores, name, hh_id, split_name="test"):
    """Calculate point-level classification/ranking metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    scores = np.asarray(scores, dtype=float)

    valid = np.isfinite(scores)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    scores = scores[valid]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = (5.0 * precision * recall) / (4.0 * precision + recall + 1e-12)
    auc_roc = _safe_auc(roc_auc_score, y_true, scores, default=0.5)
    pr_auc = _safe_auc(average_precision_score, y_true, scores, default=0.0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "detector": name,
        COL_HOUSEHOLD: hh_id,
        "split": split_name,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "f2": round(float(f2), 4),
        "pr_auc": round(float(pr_auc), 4),
        "auc_roc": round(float(auc_roc), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "support": int(y_true.sum()),
        "rows": int(len(y_true)),
    }


def _choose_f1_threshold(y_true, scores):
    """Choose a base-detector threshold from VALIDATION by maximum F1."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    y_true = y_true[valid]
    scores = scores[valid]

    if y_true.sum() == 0:
        raise ValueError("Threshold calibration requires positive validation anomalies.")

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        raise ValueError("No candidate thresholds were produced.")

    precision_t = precision[:-1]
    recall_t = recall[:-1]
    f1 = 2 * precision_t * recall_t / (precision_t + recall_t + 1e-12)
    best_f1 = np.nanmax(f1)
    candidates = np.flatnonzero(np.isclose(f1, best_f1, rtol=1e-10, atol=1e-12))

    best_idx = max(
        candidates,
        key=lambda i: (precision_t[i], recall_t[i], thresholds[i]),
    )
    threshold = float(thresholds[best_idx])
    y_pred = (scores >= threshold).astype(int)

    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": _safe_auc(average_precision_score, y_true, scores, default=0.0),
        "auc_roc": _safe_auc(roc_auc_score, y_true, scores, default=0.5),
        "support": int(y_true.sum()),
        "rows": int(len(y_true)),
    }


def _update_point_energy_features(df: pd.DataFrame, positions: np.ndarray) -> pd.DataFrame:
    """
    Recalculate only features belonging to each injected current timestamp.

    Historical lags/rolling baselines remain the clean, causal historical context.
    This prevents a point anomaly at t from contaminating later feature rows and
    being counted repeatedly as false positives.
    """
    out = df.copy()
    eps = 1e-4

    for pos in np.asarray(positions, dtype=int):
        idx = out.index[pos]
        kwh = float(out.iloc[pos][COL_KWH])

        if "rolling_24h_mean" in out.columns:
            mean24 = float(out.iloc[pos]["rolling_24h_mean"])
            out.at[idx, "deviation_24h"] = kwh - mean24
            out.at[idx, "peak_ratio"] = np.clip(kwh / max(mean24, eps), 0, 10)

        if "rolling_24h_std" in out.columns and "deviation_24h" in out.columns:
            std24 = float(out.iloc[pos]["rolling_24h_std"])
            if np.isfinite(std24) and std24 > eps:
                out.at[idx, "z_score"] = np.clip(
                    float(out.at[idx, "deviation_24h"]) / std24,
                    -10,
                    10,
                )

        if "lag_1h" in out.columns:
            lag1 = float(out.iloc[pos]["lag_1h"])
            out.at[idx, "change_1h"] = kwh - lag1
            out.at[idx, "abs_change_1h"] = abs(kwh - lag1)

        if "lag_24h" in out.columns:
            lag24 = float(out.iloc[pos]["lag_24h"])
            out.at[idx, "change_24h"] = kwh - lag24
            out.at[idx, "lag24_ratio"] = np.clip(
                (kwh + eps) / (lag24 + eps), 0, 10
            )

        if "peak_threshold_30d" in out.columns:
            threshold = float(out.iloc[pos]["peak_threshold_30d"])
            if np.isfinite(threshold):
                out.at[idx, "is_peak"] = int(kwh > threshold)

    return out


def inject_point_anomalies(df_hh: pd.DataFrame, seed: int):
    """Inject deterministic point spikes/dropouts into one clean household split."""
    df = df_hh.copy().sort_index()
    n = len(df)
    if n < 100:
        labels = pd.Series(np.zeros(n, dtype=int), index=df.index)
        types = pd.Series("normal", index=df.index, dtype=object)
        return df, labels, types

    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=np.int8)
    anomaly_type = np.full(n, "normal", dtype=object)

    valid = np.ones(n, dtype=bool)
    for col in ["rolling_24h_mean", "rolling_24h_std", "lag_1h", "lag_24h"]:
        if col in df.columns:
            valid &= np.isfinite(pd.to_numeric(df[col], errors="coerce").to_numpy())

    candidates = np.flatnonzero(valid)
    if len(candidates) == 0:
        return df, pd.Series(labels, index=df.index), pd.Series(anomaly_type, index=df.index)

    n_spikes = min(
        max(INJECTION_MIN_SPIKES, int(n * INJECTION_SPIKE_FRACTION)),
        len(candidates),
    )
    spike_idx = np.sort(rng.choice(candidates, size=n_spikes, replace=False))

    remaining = np.setdiff1d(candidates, spike_idx, assume_unique=False)
    positive = remaining[df.iloc[remaining][COL_KWH].to_numpy(dtype=float) > 0.05]
    n_dropouts = min(
        max(INJECTION_MIN_DROPOUTS, int(n * INJECTION_DROPOUT_FRACTION)),
        len(positive),
    )
    dropout_idx = (
        np.sort(rng.choice(positive, size=n_dropouts, replace=False))
        if n_dropouts > 0
        else np.asarray([], dtype=int)
    )

    kwh_col = df.columns.get_loc(COL_KWH)
    for pos in spike_idx:
        original = float(df.iloc[pos, kwh_col])
        baseline = float(df.iloc[pos].get("rolling_24h_mean", max(original, 0.1)))
        multiplier = float(rng.uniform(INJECTION_SPIKE_MIN_MULT, INJECTION_SPIKE_MAX_MULT))
        injected = max(original * multiplier, baseline * multiplier * 0.75, 0.1)
        df.iloc[pos, kwh_col] = injected
        labels[pos] = 1
        anomaly_type[pos] = "spike"

    for pos in dropout_idx:
        original = float(df.iloc[pos, kwh_col])
        df.iloc[pos, kwh_col] = max(0.0, original * INJECTION_DROPOUT_MULT)
        labels[pos] = 1
        anomaly_type[pos] = "dropout"

    changed = np.unique(np.concatenate([spike_idx, dropout_idx]))
    df = _update_point_energy_features(df, changed)

    return (
        df,
        pd.Series(labels, index=df.index, name="is_injected_anomaly"),
        pd.Series(anomaly_type, index=df.index, name="anomaly_type"),
    )


def _ae_scores_for_household(
    injected_hh: pd.DataFrame,
    history_hh: pd.DataFrame,
    ae_model,
    seq_data,
):
    from anomaly_detection.us12_autoencoder import (
        build_sequences_for_split,
        score_ae_sequences,
    )

    preprocessing = {
        "feature_cols": seq_data["feature_cols"],
        "train_medians": seq_data["train_medians"],
        "scaler": seq_data["scaler"],
    }
    X, meta = build_sequences_for_split(
        injected_hh,
        history_hh,
        preprocessing,
        max_sequences=None,
        seed=0,
    )
    if len(X) == 0:
        return pd.DataFrame()

    scored = score_ae_sequences(ae_model, X, seq_data)
    out = meta.copy()
    out["raw_anomaly_score"] = scored["recon_error"].to_numpy()
    out["anomaly_score"] = scored["anomaly_score"].to_numpy()
    return out


def calibrate_thresholds(splits: dict, if_trained: dict, ae_output: dict | None):
    """
    Calibrate base thresholds and the IF+AE ensemble on VALIDATION only.

    Multiple deterministic injection repeats are pooled to make the selected
    thresholds/weights less sensitive to one random anomaly placement.
    """
    from anomaly_detection.us11_isolation_forest import score_if_dataframe, set_if_threshold

    log_print("\n" + "=" * 70)
    log_print("US-13A: VALIDATION THRESHOLD + ENSEMBLE CALIBRATION")
    log_print("=" * 70)
    log_print(
        f"Validation injection repeats: {ENSEMBLE_CALIBRATION_REPEATS} | "
        f"ensemble objective: F{ENSEMBLE_BETA:g}"
    )

    if_y, if_raw_scores = [], []
    ae_y, ae_raw_scores = [], []
    ens_y, ens_if_scores, ens_ae_scores = [], [], []

    ae_model = ae_output["model"] if ae_output is not None else None
    seq_data = ae_output["sequence_data"] if ae_output is not None else None

    val_households = sorted(splits["val"][COL_HOUSEHOLD].unique())
    for repeat in range(int(ENSEMBLE_CALIBRATION_REPEATS)):
        for hh_id in val_households:
            val_hh = splits["val"][
                splits["val"][COL_HOUSEHOLD] == hh_id
            ].copy().sort_index()
            if len(val_hh) < 100:
                continue

            seed = (
                CALIBRATION_RANDOM_STATE
                + repeat * 100_000
                + int(hh_id) % 10_000
            )
            injected, labels, _ = inject_point_anomalies(val_hh, seed=seed)

            if_scored = score_if_dataframe(injected, if_trained)
            if_y.append(labels.to_numpy(dtype=int))
            if_raw_scores.append(if_scored["raw_anomaly_score"].to_numpy(dtype=float))

            if ae_model is None or seq_data is None:
                continue

            train_hh = splits["train"][
                splits["train"][COL_HOUSEHOLD] == hh_id
            ].copy().sort_index()
            ae_scored = _ae_scores_for_household(
                injected, train_hh, ae_model, seq_data
            )
            if ae_scored.empty:
                continue

            ae_index = pd.to_datetime(ae_scored["timestamp"])
            ae_labels = labels.reindex(ae_index).fillna(0).to_numpy(dtype=int)
            ae_y.append(ae_labels)
            ae_raw_scores.append(ae_scored["raw_anomaly_score"].to_numpy(dtype=float))

            # Ensemble calibration uses normalized 0..1 scores and rows where
            # both detector scores are available.
            aligned = pd.DataFrame(
                {
                    "timestamp": injected.index,
                    "label": labels.to_numpy(dtype=int),
                    "if_score": if_scored["anomaly_score"].to_numpy(dtype=float),
                }
            ).merge(
                ae_scored[["timestamp", "anomaly_score"]].rename(
                    columns={"anomaly_score": "ae_score"}
                ),
                on="timestamp",
                how="inner",
                validate="one_to_one",
            )
            if not aligned.empty:
                ens_y.append(aligned["label"].to_numpy(dtype=int))
                ens_if_scores.append(aligned["if_score"].to_numpy(dtype=float))
                ens_ae_scores.append(aligned["ae_score"].to_numpy(dtype=float))

    if not if_y:
        raise ValueError("No validation data available for IF threshold calibration.")

    if_cal = _choose_f1_threshold(
        np.concatenate(if_y),
        np.concatenate(if_raw_scores),
    )
    set_if_threshold(if_trained, if_cal["threshold"], source="validation_repeated_injection_max_f1")

    rows = [{"detector": "Isolation Forest", "objective": "F1", **if_cal}]
    log_print(
        "IF validation calibration: "
        f"threshold={if_cal['threshold']:.6f}, "
        f"precision={if_cal['precision']:.4f}, recall={if_cal['recall']:.4f}, "
        f"F1={if_cal['f1']:.4f}, PR-AUC={if_cal['pr_auc']:.4f}, "
        f"ROC-AUC={if_cal['auc_roc']:.4f}"
    )

    if ae_model is not None and seq_data is not None and ae_y:
        from anomaly_detection.us12_autoencoder import set_ae_threshold

        ae_cal = _choose_f1_threshold(
            np.concatenate(ae_y),
            np.concatenate(ae_raw_scores),
        )
        set_ae_threshold(seq_data, ae_cal["threshold"], source="validation_repeated_injection_max_f1")
        ae_output["threshold"] = ae_cal["threshold"]
        rows.append({"detector": "LSTM Autoencoder", "objective": "F1", **ae_cal})
        log_print(
            "AE validation calibration: "
            f"threshold={ae_cal['threshold']:.6f}, "
            f"precision={ae_cal['precision']:.4f}, recall={ae_cal['recall']:.4f}, "
            f"F1={ae_cal['f1']:.4f}, PR-AUC={ae_cal['pr_auc']:.4f}, "
            f"ROC-AUC={ae_cal['auc_roc']:.4f}"
        )

    if ens_y:
        best, grid = search_ensemble_parameters(
            np.concatenate(ens_y),
            np.concatenate(ens_if_scores),
            np.concatenate(ens_ae_scores),
            beta=ENSEMBLE_BETA,
        )
        save_ensemble_config(best, grid)
        rows.append(
            {
                "detector": "IF+AE Ensemble",
                "objective": best["objective"],
                "threshold": best["threshold"],
                "precision": best["precision"],
                "recall": best["recall"],
                "f1": best["f1"],
                "fbeta": best["fbeta"],
                "beta": best["beta"],
                "pr_auc": best["pr_auc"],
                "auc_roc": best["auc_roc"],
                "support": best["support"],
                "rows": best["rows"],
                "weight_if": best["weight_if"],
                "weight_ae": best["weight_ae"],
            }
        )
        log_print(
            "Ensemble validation calibration: "
            f"IF weight={best['weight_if']:.2f}, AE weight={best['weight_ae']:.2f}, "
            f"threshold={best['threshold']:.6f}, "
            f"precision={best['precision']:.4f}, recall={best['recall']:.4f}, "
            f"F1={best['f1']:.4f}, F{best['beta']:g}={best['fbeta']:.4f}, "
            f"PR-AUC={best['pr_auc']:.4f}, ROC-AUC={best['auc_roc']:.4f}"
        )
    elif ae_model is not None:
        raise ValueError("AE was available but no aligned IF+AE validation scores were produced.")

    calibration = pd.DataFrame(rows)
    calibration.to_csv(CALIBRATION_CSV, index=False)
    log_print(f"Saved validation calibration: {CALIBRATION_CSV}")
    return calibration


def _append_pooled(pooled, detector, y, pred, score, anomaly_type):
    pooled[detector]["y"].append(np.asarray(y, dtype=int))
    pooled[detector]["pred"].append(np.asarray(pred, dtype=int))
    pooled[detector]["score"].append(np.asarray(score, dtype=float))
    pooled[detector]["type"].append(np.asarray(anomaly_type, dtype=object))


def evaluate_on_injection(splits, if_trained, ae_output=None):
    """Evaluate locked IF, AE and ensemble settings on held-out TEST injections."""
    from anomaly_detection.us11_isolation_forest import score_if_dataframe

    log_print("\n" + "=" * 70)
    log_print("US-13B: HELD-OUT TEST INJECTION EVALUATION")
    log_print("=" * 70)

    ae_model = ae_output["model"] if ae_output is not None else None
    seq_data = ae_output["sequence_data"] if ae_output is not None else None
    ensemble_config = (
        load_ensemble_config()
        if ae_model is not None and os.path.exists(ENSEMBLE_CONFIG_PKL)
        else None
    )

    detector_names = ["Isolation Forest", "LSTM Autoencoder"]
    if ensemble_config is not None:
        detector_names.append("IF+AE Ensemble")

    detail_rows = []
    pooled = {
        name: {"y": [], "pred": [], "score": [], "type": []}
        for name in detector_names
    }

    for hh_id in sorted(splits["test"][COL_HOUSEHOLD].unique()):
        test_hh = splits["test"][
            splits["test"][COL_HOUSEHOLD] == hh_id
        ].copy().sort_index()
        if len(test_hh) < 100:
            continue

        injected, labels, anomaly_types = inject_point_anomalies(
            test_hh,
            seed=TEST_INJECTION_RANDOM_STATE + int(hh_id) % 10_000,
        )

        # Isolation Forest.
        if_scored = score_if_dataframe(injected, if_trained)
        if_raw = if_scored["raw_anomaly_score"].to_numpy(dtype=float)
        if_norm = if_scored["anomaly_score"].to_numpy(dtype=float)
        if_y = labels.to_numpy(dtype=int)
        if_types = anomaly_types.to_numpy(dtype=object)
        if_pred = (if_raw >= float(if_trained["threshold"])).astype(int)
        detail_rows.append(
            _compute_metrics(if_y, if_pred, if_raw, "Isolation Forest", hh_id, "test")
        )
        _append_pooled(pooled, "Isolation Forest", if_y, if_pred, if_raw, if_types)

        ae_scored = pd.DataFrame()
        if ae_model is not None and seq_data is not None:
            history_hh = pd.concat(
                [
                    splits["train"][splits["train"][COL_HOUSEHOLD] == hh_id],
                    splits["val"][splits["val"][COL_HOUSEHOLD] == hh_id],
                ],
                axis=0,
            ).sort_index()
            ae_scored = _ae_scores_for_household(
                injected,
                history_hh,
                ae_model,
                seq_data,
            )
            if not ae_scored.empty:
                ae_index = pd.to_datetime(ae_scored["timestamp"])
                ae_y = labels.reindex(ae_index).fillna(0).to_numpy(dtype=int)
                ae_types = anomaly_types.reindex(ae_index).fillna("normal").to_numpy(dtype=object)
                ae_raw = ae_scored["raw_anomaly_score"].to_numpy(dtype=float)
                ae_pred = (ae_raw >= float(seq_data["threshold"])).astype(int)
                detail_rows.append(
                    _compute_metrics(
                        ae_y,
                        ae_pred,
                        ae_raw,
                        "LSTM Autoencoder",
                        hh_id,
                        "test",
                    )
                )
                _append_pooled(
                    pooled,
                    "LSTM Autoencoder",
                    ae_y,
                    ae_pred,
                    ae_raw,
                    ae_types,
                )

        # Validation-selected weighted ensemble.
        if ensemble_config is not None and not ae_scored.empty:
            aligned = pd.DataFrame(
                {
                    "timestamp": injected.index,
                    "label": labels.to_numpy(dtype=int),
                    "anomaly_type": anomaly_types.to_numpy(dtype=object),
                    "if_score": if_norm,
                }
            ).merge(
                ae_scored[["timestamp", "anomaly_score"]].rename(
                    columns={"anomaly_score": "ae_score"}
                ),
                on="timestamp",
                how="inner",
                validate="one_to_one",
            )
            if not aligned.empty:
                ens_score = combine_scores(
                    aligned["if_score"].to_numpy(dtype=float),
                    aligned["ae_score"].to_numpy(dtype=float),
                    float(ensemble_config["weight_if"]),
                    float(ensemble_config["weight_ae"]),
                )
                ens_y = aligned["label"].to_numpy(dtype=int)
                ens_types = aligned["anomaly_type"].to_numpy(dtype=object)
                ens_pred = (
                    ens_score >= float(ensemble_config["threshold"])
                ).astype(int)
                detail_rows.append(
                    _compute_metrics(
                        ens_y,
                        ens_pred,
                        ens_score,
                        "IF+AE Ensemble",
                        hh_id,
                        "test",
                    )
                )
                _append_pooled(
                    pooled,
                    "IF+AE Ensemble",
                    ens_y,
                    ens_pred,
                    ens_score,
                    ens_types,
                )

    metrics = pd.DataFrame(detail_rows)
    metrics.to_csv(EVAL_CSV, index=False)

    summary_rows = []
    for detector, data in pooled.items():
        if not data["y"]:
            continue
        y = np.concatenate(data["y"])
        pred = np.concatenate(data["pred"])
        score = np.concatenate(data["score"])
        row = _compute_metrics(y, pred, score, detector, "ALL", "test")
        if detector == "IF+AE Ensemble" and ensemble_config is not None:
            row["weight_if"] = float(ensemble_config["weight_if"])
            row["weight_ae"] = float(ensemble_config["weight_ae"])
            row["decision_threshold"] = float(ensemble_config["threshold"])
            row["validation_objective"] = str(ensemble_config["objective"])
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(EVAL_SUMMARY_CSV, index=False)

    # Detection recall by injected anomaly type.  This avoids mixing the other
    # anomaly type into the negative class while still showing which events are
    # being missed.
    type_rows = []
    for detector, data in pooled.items():
        if not data["y"]:
            continue
        pred = np.concatenate(data["pred"])
        types = np.concatenate(data["type"])
        for anomaly_type in ["spike", "dropout"]:
            mask = types == anomaly_type
            total = int(mask.sum())
            detected = int(pred[mask].sum()) if total else 0
            type_rows.append(
                {
                    "detector": detector,
                    "anomaly_type": anomaly_type,
                    "detected": detected,
                    "total": total,
                    "recall": (detected / total) if total else np.nan,
                }
            )
    by_type = pd.DataFrame(type_rows)
    by_type.to_csv(EVAL_BY_TYPE_CSV, index=False)

    _print_aggregate(metrics, summary)
    _print_by_type(by_type)
    plot_evaluation_chart(metrics)
    log_print(f"Saved detailed metrics: {EVAL_CSV}")
    log_print(f"Saved pooled summary: {EVAL_SUMMARY_CSV}")
    log_print(f"Saved anomaly-type detection: {EVAL_BY_TYPE_CSV}")
    return metrics, summary


def _print_aggregate(detail: pd.DataFrame, summary: pd.DataFrame):
    log_print("\nAGGREGATE TEST METRICS")
    order = ["Isolation Forest", "LSTM Autoencoder", "IF+AE Ensemble"]
    for detector in order:
        sub = detail[detail["detector"] == detector]
        if sub.empty:
            continue
        log_print(detector)
        for metric in ["precision", "recall", "f1", "f2", "pr_auc", "auc_roc"]:
            log_print(
                f"  macro {metric}: {sub[metric].mean():.4f} +/- "
                f"{sub[metric].std(ddof=0):.4f}"
            )
        pooled = summary[summary["detector"] == detector]
        if not pooled.empty:
            r = pooled.iloc[0]
            log_print(
                f"  pooled: precision={r['precision']:.4f}, recall={r['recall']:.4f}, "
                f"F1={r['f1']:.4f}, F2={r['f2']:.4f}, PR-AUC={r['pr_auc']:.4f}, "
                f"ROC-AUC={r['auc_roc']:.4f} | "
                f"TP={int(r['tp'])}, FP={int(r['fp'])}, FN={int(r['fn'])}, TN={int(r['tn'])}"
            )
            if detector == "IF+AE Ensemble" and "weight_if" in pooled.columns:
                log_print(
                    f"  locked ensemble: IF={float(r['weight_if']):.2f}, "
                    f"AE={float(r['weight_ae']):.2f}, "
                    f"threshold={float(r['decision_threshold']):.4f}"
                )


def _print_by_type(by_type: pd.DataFrame):
    if by_type.empty:
        return
    log_print("\nTEST DETECTION BY INJECTED ANOMALY TYPE")
    for detector in by_type["detector"].unique():
        log_print(detector)
        sub = by_type[by_type["detector"] == detector]
        for _, row in sub.iterrows():
            log_print(
                f"  {row['anomaly_type']}: {int(row['detected'])}/{int(row['total'])} "
                f"detected (recall={float(row['recall']):.4f})"
            )


def plot_evaluation_chart(df: pd.DataFrame):
    if df.empty:
        return

    metric_names = ["precision", "recall", "f1", "pr_auc"]
    detector_order = ["Isolation Forest", "LSTM Autoencoder", "IF+AE Ensemble"]
    detectors = [d for d in detector_order if d in set(df["detector"])]
    hh_ids = sorted(df[COL_HOUSEHOLD].dropna().unique())
    x = np.arange(len(hh_ids))
    width = 0.8 / max(1, len(detectors))

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for ax, metric in zip(axes.flat, metric_names):
        for i, detector in enumerate(detectors):
            sub = df[df["detector"] == detector]
            vals = []
            for hh in hh_ids:
                hit = sub[sub[COL_HOUSEHOLD] == hh]
                vals.append(float(hit[metric].iloc[0]) if not hit.empty else 0.0)
            ax.bar(
                x + (i - (len(detectors) - 1) / 2) * width,
                vals,
                width,
                label=detector,
            )
        ax.set_title(metric.upper())
        ax.set_xticks(x)
        ax.set_xticklabels([f"HH{i + 1}" for i in range(len(hh_ids))], rotation=45)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2, axis="y")
        ax.legend(fontsize=8)

    plt.tight_layout()
    safe_savefig(fig, EVAL_GRAPH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def verify_us13(metrics: pd.DataFrame, summary: pd.DataFrame | None = None) -> bool:
    detectors = set(metrics.get("detector", pd.Series(dtype=str)).dropna().values)
    expected = {"Isolation Forest", "LSTM Autoencoder", "IF+AE Ensemble"}
    checks = [
        ("Isolation Forest evaluated", "Isolation Forest" in detectors),
        ("LSTM Autoencoder evaluated", "LSTM Autoencoder" in detectors),
        ("IF+AE Ensemble evaluated", "IF+AE Ensemble" in detectors),
        ("detailed metrics saved", os.path.exists(EVAL_CSV)),
        ("pooled summary saved", os.path.exists(EVAL_SUMMARY_CSV)),
        ("anomaly-type metrics saved", os.path.exists(EVAL_BY_TYPE_CSV)),
        ("ensemble calibration saved", os.path.exists(ENSEMBLE_CONFIG_PKL)),
        ("PR-AUC reported", "pr_auc" in metrics.columns),
    ]
    if summary is not None:
        summary_detectors = set(summary.get("detector", pd.Series(dtype=str)).dropna().values)
        checks.append(("all pooled detectors reported", expected.issubset(summary_detectors)))

    ok = all(value for _, value in checks)
    for name, passed in checks:
        log_print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    log_print(f"US-13 status: {'PASS' if ok else 'FAIL'}")
    return bool(ok)
