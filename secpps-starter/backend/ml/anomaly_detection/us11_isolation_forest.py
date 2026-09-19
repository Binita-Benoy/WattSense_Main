from logging_utils import log_print
import logging
logger = logging.getLogger()
# US-11: Isolation Forest using energy, temporal and weather/context features.
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from config import (
    IF_CONTAMINATION, IF_N_ESTIMATORS, IF_RANDOM_STATE,
    IF_MODEL_PKL, IF_SCALER_PKL, IF_RESULTS_CSV, IF_GRAPH,
    COL_HOUSEHOLD, COL_KWH, PRIMARY_HH, SEVERITY_HIGH, SEVERITY_MEDIUM
)

# Deliberately include different signal families, not only energy-derived features.
IF_FEATURES = [
    # Energy / temporal behaviour
    COL_KWH, "lag_1h", "lag_24h", "lag_48h", "lag_168h",
    "rolling_24h_mean", "rolling_24h_std", "rolling_7d_mean", "rolling_7d_std",
    "rolling_30d_mean", "rolling_30d_std", "deviation_24h", "z_score",
    "change_1h", "change_24h", "abs_change_1h", "peak_ratio",
    # Calendar context
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_peak", "is_weekend", "season",
    # Weather context
    "temperature", "humidity", "apparent_temperature", "pressure", "wind_speed",
    "cloud_cover", "precip_intensity", "dew_point", "precip_probability",
    "shortwave_radiation", "temperature_24h_mean", "temperature_24h_std",
    "temperature_deviation_24h", "humidity_24h_mean", "humidity_deviation_24h",
    "heating_degree", "cooling_degree", "solar_generation_context"
]


def train_isolation_forest(F_train):
    log_print("\n" + "=" * 70)
    log_print("US-11: ISOLATION FOREST")
    log_print("=" * 70)
    available = [c for c in IF_FEATURES if c in F_train.columns]
    missing = [c for c in IF_FEATURES if c not in F_train.columns]
    if missing:
        log_print(f"Missing/skipped features: {missing}")
    X = F_train[available].astype(float).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_scaled)
    joblib.dump(model, IF_MODEL_PKL)
    joblib.dump(scaler, IF_SCALER_PKL)
    log_print(f"Training rows: {len(F_train):,}")
    log_print(f"Features used: {len(available)}")
    log_print(f"Features: {available}")
    log_print(f"Saved model: {IF_MODEL_PKL}")
    return {"model": model, "scaler": scaler, "feature_cols": available}


def predict_if_on_split(F_split, trained, split_name="test"):
    cols = trained["feature_cols"]
    X = F_split[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X_scaled = trained["scaler"].transform(X)
    model = trained["model"]
    raw = model.predict(X_scaled)
    raw_scores = model.score_samples(X_scaled)
    score = 1 - ((raw_scores - raw_scores.min()) /
                 (raw_scores.max() - raw_scores.min() + 1e-9))
    result = F_split[[COL_HOUSEHOLD, COL_KWH]].copy()
    result = result.rename(columns={COL_KWH: "consumption"})
    result["anomaly_score"] = np.clip(score, 0, 1)
    result["is_anomaly"] = (raw == -1).astype(int)
    result["split"] = split_name
    return result


def run_if_pipeline(splits):
    trained = train_isolation_forest(splits["train"])
    results_val = predict_if_on_split(splits["val"], trained, "val")
    results_test = predict_if_on_split(splits["test"], trained, "test")
    results_test.to_csv(IF_RESULTS_CSV)
    log_print(f"Test anomalies: {results_test['is_anomaly'].sum():,} / {len(results_test):,}")
    log_print(f"Saved: {IF_RESULTS_CSV}")
    plot_if_evidence(results_test)
    verify_us11(results_test)
    return {"trained": trained, "results_val": results_val, "results_test": results_test}


def plot_if_evidence(results_test, household_id=PRIMARY_HH):
    df = results_test[results_test[COL_HOUSEHOLD] == household_id].copy()
    if df.empty:
        return
    df.index = pd.to_datetime(df.index)
    normal = df[df["is_anomaly"] == 0]
    anomalies = df[df["is_anomaly"] == 1]
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    axes[0].plot(normal.index, normal["consumption"], lw=0.6, label="Normal")
    axes[0].scatter(anomalies.index, anomalies["consumption"], s=18, label="Anomaly")
    axes[0].set_ylabel("kWh")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(df.index, df["anomaly_score"], lw=0.7, label="IF score")
    axes[1].axhline(SEVERITY_MEDIUM, ls="--", lw=1, label="Medium")
    axes[1].axhline(SEVERITY_HIGH, ls="--", lw=1, label="High")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Anomaly score")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b '%y"))
    plt.tight_layout()
    plt.savefig(IF_GRAPH, dpi=150, bbox_inches="tight")
    plt.close()


def verify_us11(results):
    checks = [
        ("timestamp index", isinstance(results.index, pd.DatetimeIndex)),
        ("consumption", "consumption" in results.columns),
        ("anomaly score", "anomaly_score" in results.columns),
        ("is_anomaly", "is_anomaly" in results.columns),
        ("results saved", os.path.exists(IF_RESULTS_CSV)),
    ]
    ok = all(v for _, v in checks)
    log_print(f"US-11 status: {'PASS' if ok else 'FAIL'}")
    return ok
