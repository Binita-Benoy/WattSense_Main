"""US-11: Isolation Forest anomaly detector with leakage-safe preprocessing."""
from __future__ import annotations

import logging
import os

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from config import (
    IF_CONTAMINATION,
    IF_N_ESTIMATORS,
    IF_MAX_SAMPLES,
    IF_MAX_FEATURES,
    IF_RANDOM_STATE,
    IF_FALLBACK_THRESHOLD_PCT,
    IF_SCORE_LOW_PCT,
    IF_SCORE_HIGH_PCT,
    IF_MODEL_PKL,
    IF_SCALER_PKL,
    IF_PREPROCESS_PKL,
    IF_THRESHOLD_PKL,
    IF_RESULTS_CSV,
    IF_GRAPH,
    COL_HOUSEHOLD,
    COL_KWH,
    COL_PUBLIC_HOLIDAY,
    PRIMARY_HH,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
)
from logging_utils import log_print, safe_savefig

logger = logging.getLogger()

# Focused feature set: keep strong energy/context signals and avoid unnecessarily
# diluting Isolation Forest splits with many highly correlated covariates.
IF_FEATURES = [
    # Current and historical energy behaviour.
    COL_KWH,
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_24h_mean", "rolling_24h_std",
    "rolling_7d_mean", "rolling_7d_std",
    "deviation_24h", "z_score",
    "change_1h", "change_24h", "abs_change_1h",
    "peak_ratio", "lag24_ratio",
    # Calendar context.
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", COL_PUBLIC_HOLIDAY,
    # Weather/context features most relevant to residential demand.
    "temperature", "apparent_temperature", "humidity",
    "precip_intensity", "shortwave_radiation",
    "temperature_deviation_24h", "heating_degree", "cooling_degree",
]


def _normalise_scores(raw_scores: np.ndarray, low: float, high: float) -> np.ndarray:
    """Map raw IF anomaly scores to a stable train-referenced 0..1 scale."""
    denom = max(float(high) - float(low), 1e-9)
    return np.clip((raw_scores - float(low)) / denom, 0.0, 1.0)


def _prepare_prediction_matrix(F: pd.DataFrame, trained: dict) -> pd.DataFrame:
    cols = trained["feature_cols"]
    # reindex creates any context columns that are absent at inference time
    # (for example weather for a brand-new household). They are then filled
    # using TRAIN-only medians saved with the model.
    X = F.reindex(columns=cols).astype(float).replace([np.inf, -np.inf], np.nan)
    return X.fillna(trained["train_medians"])


def train_isolation_forest(F_train: pd.DataFrame) -> dict:
    """Fit IF only on the chronological training split."""
    log_print("\n" + "=" * 70)
    log_print("US-11: ISOLATION FOREST")
    log_print("=" * 70)

    available = [c for c in IF_FEATURES if c in F_train.columns]
    missing = [c for c in IF_FEATURES if c not in F_train.columns]
    if missing:
        log_print(f"Missing/skipped IF features: {missing}")
    if not available:
        raise ValueError("No Isolation Forest features are available.")

    X = F_train[available].astype(float).replace([np.inf, -np.inf], np.nan)
    train_medians = X.median().fillna(0.0)
    X = X.fillna(train_medians)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    max_samples = min(int(IF_MAX_SAMPLES), len(X))
    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        max_samples=max_samples,
        max_features=IF_MAX_FEATURES,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # sklearn score_samples: smaller = more abnormal. Negating makes larger =
    # more anomalous, which simplifies thresholding and interpretation.
    train_raw_scores = -model.score_samples(X_scaled)
    score_low = float(np.percentile(train_raw_scores, IF_SCORE_LOW_PCT))
    score_high = float(np.percentile(train_raw_scores, IF_SCORE_HIGH_PCT))
    fallback_threshold = float(
        np.percentile(train_raw_scores, IF_FALLBACK_THRESHOLD_PCT)
    )

    trained = {
        "model": model,
        "scaler": scaler,
        "feature_cols": available,
        "train_medians": train_medians,
        "score_low": score_low,
        "score_high": score_high,
        "threshold": fallback_threshold,
        "threshold_source": f"train_{IF_FALLBACK_THRESHOLD_PCT:g}th_percentile_fallback",
    }

    joblib.dump(model, IF_MODEL_PKL)
    joblib.dump(scaler, IF_SCALER_PKL)
    joblib.dump(
        {
            "feature_cols": available,
            "train_medians": train_medians,
            "score_low": score_low,
            "score_high": score_high,
            "threshold": fallback_threshold,
            "threshold_source": trained["threshold_source"],
        },
        IF_PREPROCESS_PKL,
    )
    joblib.dump(fallback_threshold, IF_THRESHOLD_PKL)

    log_print(f"Training rows: {len(F_train):,}")
    log_print(f"Features used ({len(available)}): {available}")
    log_print(
        f"IF settings: trees={IF_N_ESTIMATORS}, max_samples={max_samples}, "
        f"max_features={IF_MAX_FEATURES}, contamination={IF_CONTAMINATION}"
    )
    log_print(f"Fallback raw-score threshold: {fallback_threshold:.6f}")
    log_print(f"Saved model: {IF_MODEL_PKL}")
    return trained


def score_if_dataframe(F_split: pd.DataFrame, trained: dict) -> pd.DataFrame:
    """Return raw and normalized IF scores without making threshold decisions."""
    if F_split.empty:
        return pd.DataFrame(index=F_split.index)

    X = _prepare_prediction_matrix(F_split, trained)
    X_scaled = trained["scaler"].transform(X)
    raw_scores = -trained["model"].score_samples(X_scaled)
    norm_scores = _normalise_scores(
        raw_scores, trained["score_low"], trained["score_high"]
    )

    return pd.DataFrame(
        {
            "raw_anomaly_score": raw_scores,
            "anomaly_score": norm_scores,
        },
        index=F_split.index,
    )


def set_if_threshold(trained: dict, threshold: float, source: str = "validation_f1") -> dict:
    """Set and persist a calibrated raw anomaly-score threshold."""
    trained["threshold"] = float(threshold)
    trained["threshold_source"] = source
    joblib.dump(float(threshold), IF_THRESHOLD_PKL)

    preprocess = {
        "feature_cols": trained["feature_cols"],
        "train_medians": trained["train_medians"],
        "score_low": trained["score_low"],
        "score_high": trained["score_high"],
        "threshold": float(threshold),
        "threshold_source": source,
    }
    joblib.dump(preprocess, IF_PREPROCESS_PKL)
    log_print(f"IF threshold set to {threshold:.6f} ({source})")
    return trained


def predict_if_on_split(
    F_split: pd.DataFrame,
    trained: dict,
    split_name: str = "test",
    threshold: float | None = None,
) -> pd.DataFrame:
    """Predict anomalies using the calibrated raw-score threshold."""
    if F_split.empty:
        return pd.DataFrame()

    threshold = float(trained["threshold"] if threshold is None else threshold)
    scores = score_if_dataframe(F_split, trained)

    result = F_split[[COL_HOUSEHOLD, COL_KWH]].copy()
    result = result.rename(columns={COL_KWH: "consumption"})
    result["raw_anomaly_score"] = scores["raw_anomaly_score"].to_numpy()
    result["anomaly_score"] = scores["anomaly_score"].to_numpy()
    result["is_anomaly"] = (
        result["raw_anomaly_score"].to_numpy() >= threshold
    ).astype("int8")
    result["decision_threshold"] = threshold
    result["split"] = split_name
    return result


def run_if_pipeline(splits: dict, threshold: float | None = None) -> dict:
    """
    Standalone US-11 pipeline.

    In the full runner, US-13 calibrates the threshold on VALIDATION and then
    final IF predictions are regenerated. If called alone, the train-percentile
    fallback threshold is used.
    """
    trained = train_isolation_forest(splits["train"])
    if threshold is not None:
        set_if_threshold(trained, threshold, "provided")

    results_val = predict_if_on_split(splits["val"], trained, "val")
    results_test = predict_if_on_split(splits["test"], trained, "test")
    results_test.to_csv(IF_RESULTS_CSV)

    log_print(
        f"Test anomalies: {int(results_test['is_anomaly'].sum()):,} / "
        f"{len(results_test):,}"
    )
    log_print(f"Saved: {IF_RESULTS_CSV}")
    plot_if_evidence(results_test, trained)
    verify_us11(results_test)
    return {
        "trained": trained,
        "results_val": results_val,
        "results_test": results_test,
    }


def plot_if_evidence(
    results_test: pd.DataFrame,
    trained: dict | None = None,
    household_id: int = PRIMARY_HH,
) -> None:
    if results_test.empty:
        return

    df = results_test[results_test[COL_HOUSEHOLD] == household_id].copy()
    if df.empty:
        # Use first available household if PRIMARY_HH is absent.
        household_id = int(results_test[COL_HOUSEHOLD].iloc[0])
        df = results_test[results_test[COL_HOUSEHOLD] == household_id].copy()

    df.index = pd.to_datetime(df.index)
    normal = df[df["is_anomaly"] == 0]
    anomalies = df[df["is_anomaly"] == 1]

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    fig.suptitle(f"US-11 Isolation Forest — Household {household_id}", fontsize=12)

    axes[0].plot(df.index, df["consumption"], lw=0.55, alpha=0.75, label="Consumption")
    axes[0].scatter(
        anomalies.index,
        anomalies["consumption"],
        s=20,
        zorder=5,
        label=f"Detected anomaly ({len(anomalies)})",
    )
    axes[0].set_ylabel("kWh")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(df.index, df["anomaly_score"], lw=0.7, label="IF anomaly score")
    axes[1].axhline(SEVERITY_MEDIUM, ls="--", lw=1, label="Medium severity")
    axes[1].axhline(SEVERITY_HIGH, ls="--", lw=1, label="High severity")

    if trained is not None:
        threshold_norm = float(
            _normalise_scores(
                np.asarray([trained["threshold"]]),
                trained["score_low"],
                trained["score_high"],
            )[0]
        )
        axes[1].axhline(
            threshold_norm,
            ls=":",
            lw=1.4,
            label=f"Detection threshold ({threshold_norm:.2f})",
        )

    axes[1].scatter(
        anomalies.index,
        anomalies["anomaly_score"],
        s=14,
        zorder=5,
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Anomaly score")
    axes[1].set_xlabel("Timestamp")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b '%y"))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    safe_savefig(fig, IF_GRAPH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def verify_us11(results: pd.DataFrame) -> bool:
    checks = [
        ("timestamp index", isinstance(results.index, pd.DatetimeIndex)),
        ("consumption", "consumption" in results.columns),
        ("anomaly score", "anomaly_score" in results.columns),
        ("raw anomaly score", "raw_anomaly_score" in results.columns),
        ("is_anomaly", "is_anomaly" in results.columns),
        ("results saved", os.path.exists(IF_RESULTS_CSV)),
    ]
    ok = all(value for _, value in checks)
    for name, passed in checks:
        log_print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    log_print(f"US-11 status: {'PASS' if ok else 'FAIL'}")
    return bool(ok)
