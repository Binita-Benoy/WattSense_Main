"""Validation-calibrated weighted ensemble for Isolation Forest + LSTM AE.

The ensemble operates on the stable 0..1 anomaly scores produced by the two
base detectors.  A validation-only grid search selects the AE/IF weight and a
single decision threshold.  The default objective is F2, which places more
weight on recall than precision and is therefore appropriate when the project
priority is detecting more true anomalies while still penalising false alarms.

TEST data is never used when choosing the weights or threshold.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable

import joblib
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
    COL_HOUSEHOLD,
    ENSEMBLE_BETA,
    ENSEMBLE_CONFIG_PKL,
    ENSEMBLE_GRAPH,
    ENSEMBLE_GRID_CSV,
    ENSEMBLE_RESULTS_CSV,
    ENSEMBLE_WEIGHT_STEP,
)
from logging_utils import log_print, safe_savefig

logger = logging.getLogger()


def _safe_auc(metric_fn, y_true, scores, default: float) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float(default)
        return float(metric_fn(y_true, scores))
    except Exception:
        return float(default)


def _fbeta(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    beta2 = float(beta) ** 2
    return (
        (1.0 + beta2) * precision * recall
        / (beta2 * precision + recall + 1e-12)
    )


def _choose_fbeta_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    beta: float = ENSEMBLE_BETA,
) -> dict:
    """Choose a threshold that maximises F-beta on VALIDATION only."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    mask = np.isfinite(scores)
    y_true = y_true[mask]
    scores = scores[mask]

    if len(y_true) == 0 or int(y_true.sum()) == 0:
        raise ValueError("Ensemble calibration requires positive validation anomalies.")

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        raise ValueError("No ensemble threshold candidates were produced.")

    precision_t = precision[:-1]
    recall_t = recall[:-1]
    fbeta = _fbeta(precision_t, recall_t, beta)
    best = float(np.nanmax(fbeta))
    candidates = np.flatnonzero(np.isclose(fbeta, best, rtol=1e-10, atol=1e-12))

    # Detection-focused tie-break: recall first, then precision.  If still tied,
    # prefer the larger threshold to avoid unnecessary false positives.
    best_idx = max(
        candidates,
        key=lambda i: (recall_t[i], precision_t[i], thresholds[i]),
    )

    threshold = float(thresholds[best_idx])
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "fbeta": float(best),
        "beta": float(beta),
        "pr_auc": _safe_auc(average_precision_score, y_true, scores, 0.0),
        "auc_roc": _safe_auc(roc_auc_score, y_true, scores, 0.5),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "support": int(y_true.sum()),
        "rows": int(len(y_true)),
    }


def combine_scores(
    if_scores: Iterable[float],
    ae_scores: Iterable[float],
    weight_if: float,
    weight_ae: float,
) -> np.ndarray:
    """Weighted 0..1 score with graceful fallback if one detector is missing."""
    if_scores = np.asarray(if_scores, dtype=float)
    ae_scores = np.asarray(ae_scores, dtype=float)
    if if_scores.shape != ae_scores.shape:
        raise ValueError("IF and AE score arrays must have the same shape.")

    if_ok = np.isfinite(if_scores)
    ae_ok = np.isfinite(ae_scores)

    numerator = np.zeros_like(if_scores, dtype=float)
    denominator = np.zeros_like(if_scores, dtype=float)

    numerator += np.where(if_ok, float(weight_if) * if_scores, 0.0)
    denominator += np.where(if_ok, float(weight_if), 0.0)

    numerator += np.where(ae_ok, float(weight_ae) * ae_scores, 0.0)
    denominator += np.where(ae_ok, float(weight_ae), 0.0)

    out = np.full_like(if_scores, np.nan, dtype=float)
    valid = denominator > 0
    out[valid] = numerator[valid] / denominator[valid]
    return np.clip(out, 0.0, 1.0)


def search_ensemble_parameters(
    y_true,
    if_scores,
    ae_scores,
    beta: float = ENSEMBLE_BETA,
    weight_step: float = ENSEMBLE_WEIGHT_STEP,
):
    """
    Grid-search IF/AE weights and the decision threshold on VALIDATION.

    Returns
    -------
    best_config : dict
        Selected weights + threshold + validation metrics.
    grid : DataFrame
        One row per candidate AE weight for auditability.
    """
    y_true = np.asarray(y_true, dtype=int)
    if_scores = np.asarray(if_scores, dtype=float)
    ae_scores = np.asarray(ae_scores, dtype=float)

    if not (len(y_true) == len(if_scores) == len(ae_scores)):
        raise ValueError("Ensemble calibration arrays must have equal length.")

    both = np.isfinite(if_scores) & np.isfinite(ae_scores)
    y = y_true[both]
    if_s = np.clip(if_scores[both], 0.0, 1.0)
    ae_s = np.clip(ae_scores[both], 0.0, 1.0)

    if len(y) == 0:
        raise ValueError("No rows contain both IF and AE validation scores.")

    step = float(weight_step)
    if not (0 < step <= 1):
        raise ValueError("ENSEMBLE_WEIGHT_STEP must be in (0, 1].")

    n_steps = int(round(1.0 / step))
    weights_ae = np.linspace(0.0, 1.0, n_steps + 1)
    rows = []

    for w_ae in weights_ae:
        w_ae = float(w_ae)
        w_if = float(1.0 - w_ae)
        score = w_if * if_s + w_ae * ae_s
        result = _choose_fbeta_threshold(y, score, beta=beta)
        rows.append(
            {
                "weight_if": w_if,
                "weight_ae": w_ae,
                **result,
            }
        )

    grid = pd.DataFrame(rows)

    # Main objective: F-beta.  For ties prefer higher recall, then precision,
    # then PR-AUC.  The final tie-break favours AE because temporal information
    # is generally complementary to point-wise IF evidence; it is reached only
    # for numerically identical validation results.
    ranking = grid.sort_values(
        ["fbeta", "recall", "precision", "pr_auc", "weight_ae"],
        ascending=[False, False, False, False, False],
        kind="mergesort",
    )
    best = ranking.iloc[0].to_dict()
    best["objective"] = f"F{float(beta):g}"
    best["method"] = "weighted_sum"
    best["score_space"] = "train_referenced_0_1"
    return best, grid


def save_ensemble_config(config: dict, grid: pd.DataFrame) -> None:
    """Persist the validation-selected ensemble configuration and grid."""
    joblib.dump(config, ENSEMBLE_CONFIG_PKL)
    grid.to_csv(ENSEMBLE_GRID_CSV, index=False)
    log_print(f"Saved ensemble configuration: {ENSEMBLE_CONFIG_PKL}")
    log_print(f"Saved ensemble weight search: {ENSEMBLE_GRID_CSV}")


def load_ensemble_config() -> dict:
    if not os.path.exists(ENSEMBLE_CONFIG_PKL):
        raise FileNotFoundError(
            f"Ensemble configuration not found: {ENSEMBLE_CONFIG_PKL}. "
            "Run validation calibration first."
        )
    return joblib.load(ENSEMBLE_CONFIG_PKL)


def _normalise_if_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" not in out.columns:
        out = out.reset_index()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out


def _normalise_ae_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out


def predict_ensemble_from_results(
    if_results: pd.DataFrame,
    ae_results: pd.DataFrame,
    config: dict,
    split_name: str = "test",
) -> pd.DataFrame:
    """Build final ensemble predictions from already-scored base detectors."""
    df_if = _normalise_if_results(if_results).rename(
        columns={
            "anomaly_score": "if_score",
            "raw_anomaly_score": "if_raw_score",
            "is_anomaly": "if_flag",
        }
    )
    df_ae = _normalise_ae_results(ae_results).rename(
        columns={
            "anomaly_score": "ae_score",
            "recon_error": "ae_raw_score",
            "is_anomaly": "ae_flag",
        }
    )

    key = [COL_HOUSEHOLD, "timestamp"]
    left_cols = key + ["consumption", "if_score", "if_raw_score", "if_flag"]
    right_cols = key + ["consumption", "ae_score", "ae_raw_score", "ae_flag"]

    left = df_if[left_cols].rename(columns={"consumption": "consumption_if"})
    right = df_ae[right_cols].rename(columns={"consumption": "consumption_ae"})
    df = left.merge(right, on=key, how="outer", validate="one_to_one")

    df["consumption"] = df["consumption_if"].combine_first(df["consumption_ae"])
    df["if_score"] = pd.to_numeric(df["if_score"], errors="coerce")
    df["ae_score"] = pd.to_numeric(df["ae_score"], errors="coerce")
    df["if_flag"] = df["if_flag"].fillna(0).astype(int)
    df["ae_flag"] = df["ae_flag"].fillna(0).astype(int)

    w_if = float(config["weight_if"])
    w_ae = float(config["weight_ae"])
    threshold = float(config["threshold"])

    df["ensemble_score"] = combine_scores(
        df["if_score"].to_numpy(),
        df["ae_score"].to_numpy(),
        w_if,
        w_ae,
    )
    df["is_anomaly"] = (
        np.isfinite(df["ensemble_score"].to_numpy())
        & (df["ensemble_score"].to_numpy() >= threshold)
    ).astype("int8")

    df["decision_threshold"] = threshold
    df["weight_if"] = w_if
    df["weight_ae"] = w_ae
    df["split"] = split_name
    df["has_both_scores"] = (df["if_score"].notna() & df["ae_score"].notna()).astype("int8")

    both = (df["if_flag"] == 1) & (df["ae_flag"] == 1)
    if_only = (df["if_flag"] == 1) & (df["ae_flag"] == 0)
    ae_only = (df["if_flag"] == 0) & (df["ae_flag"] == 1)
    df["base_detector_agreement"] = np.select(
        [both, if_only, ae_only],
        ["both", "if_only", "ae_only"],
        default="neither",
    )

    ordered = [
        "timestamp",
        COL_HOUSEHOLD,
        "consumption",
        "ensemble_score",
        "is_anomaly",
        "decision_threshold",
        "weight_if",
        "weight_ae",
        "if_score",
        "ae_score",
        "if_raw_score",
        "ae_raw_score",
        "if_flag",
        "ae_flag",
        "base_detector_agreement",
        "has_both_scores",
        "split",
    ]
    return (
        df[ordered]
        .sort_values([COL_HOUSEHOLD, "timestamp"])
        .reset_index(drop=True)
    )


def save_ensemble_results(results: pd.DataFrame) -> None:
    results.to_csv(ENSEMBLE_RESULTS_CSV, index=False)
    log_print(f"Saved ensemble results: {ENSEMBLE_RESULTS_CSV}")


def plot_ensemble_evidence(results: pd.DataFrame, config: dict) -> None:
    if results.empty:
        return

    hh = int(results[COL_HOUSEHOLD].iloc[0])
    df = results[results[COL_HOUSEHOLD] == hh].copy().sort_values("timestamp")
    anomalies = df[df["is_anomaly"] == 1]
    threshold = float(config["threshold"])

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    fig.suptitle(
        f"Validation-calibrated IF + LSTM ensemble — Household {hh}",
        fontsize=12,
    )

    axes[0].plot(df["timestamp"], df["consumption"], lw=0.55, label="Consumption")
    axes[0].scatter(
        anomalies["timestamp"],
        anomalies["consumption"],
        s=20,
        zorder=5,
        label=f"Ensemble anomaly ({len(anomalies)})",
    )
    axes[0].set_ylabel("kWh")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(df["timestamp"], df["if_score"], lw=0.6, alpha=0.7, label="IF score")
    axes[1].plot(df["timestamp"], df["ae_score"], lw=0.6, alpha=0.7, label="AE score")
    axes[1].plot(df["timestamp"], df["ensemble_score"], lw=1.0, label="Ensemble score")
    axes[1].axhline(threshold, ls="--", lw=1.2, label=f"Ensemble threshold ({threshold:.3f})")
    axes[1].scatter(
        anomalies["timestamp"],
        anomalies["ensemble_score"],
        s=14,
        zorder=5,
    )
    axes[1].set_ylabel("Anomaly score (0–1)")
    axes[1].set_xlabel("Timestamp")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(ncol=2)
    axes[1].grid(alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    safe_savefig(fig, ENSEMBLE_GRAPH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def verify_ensemble(results: pd.DataFrame) -> bool:
    checks = [
        ("ensemble results not empty", not results.empty),
        ("ensemble score", "ensemble_score" in results.columns),
        ("ensemble anomaly flag", "is_anomaly" in results.columns),
        ("IF score retained", "if_score" in results.columns),
        ("AE score retained", "ae_score" in results.columns),
        ("calibration saved", os.path.exists(ENSEMBLE_CONFIG_PKL)),
        ("results saved", os.path.exists(ENSEMBLE_RESULTS_CSV)),
    ]
    ok = all(value for _, value in checks)
    for name, passed in checks:
        log_print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    log_print(f"Ensemble status: {'PASS' if ok else 'FAIL'}")
    return bool(ok)
