"""US-14: dashboard/API output using the validation-calibrated IF+AE ensemble."""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd

from config import (
    API_CSV,
    API_JSON,
    COL_HOUSEHOLD,
    ENSEMBLE_RESULTS_CSV,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
)
from logging_utils import log_print

logger = logging.getLogger()


def assign_severity(score: float, is_anomaly: int) -> str:
    if int(is_anomaly) == 0:
        return "NORMAL"
    if float(score) >= SEVERITY_HIGH:
        return "HIGH"
    if float(score) >= SEVERITY_MEDIUM:
        return "MEDIUM"
    return "LOW"


def build_api_payload() -> pd.DataFrame:
    """
    Build dashboard rows from the final ensemble output.

    The ensemble decision is authoritative.  Individual IF/AE flags and scores
    are retained as model evidence so the dashboard can explain whether the two
    base detectors agreed with the final ensemble alert.
    """
    log_print("\n" + "=" * 70)
    log_print("US-14: BUILDING ENSEMBLE API PAYLOAD")
    log_print("=" * 70)

    if not os.path.exists(ENSEMBLE_RESULTS_CSV):
        raise FileNotFoundError(
            f"Ensemble results not found: {ENSEMBLE_RESULTS_CSV}. "
            "Run run_anomaly_detection.py first."
        )

    df = pd.read_csv(ENSEMBLE_RESULTS_CSV, parse_dates=["timestamp"])
    if df.empty:
        raise ValueError("Ensemble results are empty.")

    df["anomaly_score"] = pd.to_numeric(
        df["ensemble_score"], errors="coerce"
    ).fillna(0.0).clip(0, 1)
    df["is_anomaly"] = pd.to_numeric(
        df["is_anomaly"], errors="coerce"
    ).fillna(0).astype(int)
    df["if_flag"] = pd.to_numeric(df["if_flag"], errors="coerce").fillna(0).astype(int)
    df["ae_flag"] = pd.to_numeric(df["ae_flag"], errors="coerce").fillna(0).astype(int)
    df["if_score"] = pd.to_numeric(df["if_score"], errors="coerce").fillna(0.0)
    df["ae_score"] = pd.to_numeric(df["ae_score"], errors="coerce").fillna(0.0)

    df["severity"] = [
        assign_severity(score, flag)
        for score, flag in zip(df["anomaly_score"], df["is_anomaly"])
    ]

    both = (df["if_flag"] == 1) & (df["ae_flag"] == 1)
    either = (df["if_flag"] == 1) | (df["ae_flag"] == 1)
    alert = df["is_anomaly"] == 1

    # Confidence describes base-model agreement with an ensemble alert.  An
    # ensemble may legitimately cross its weighted threshold even when neither
    # standalone model crosses its own independently calibrated threshold.
    df["confidence"] = np.select(
        [alert & both, alert & either, alert],
        ["HIGH", "MEDIUM", "LOW"],
        default="NORMAL",
    )

    evidence = np.full(len(df), "none", dtype=object)
    evidence[(alert & (df["if_flag"] == 1) & (df["ae_flag"] == 0)).to_numpy()] = "if_only"
    evidence[(alert & (df["if_flag"] == 0) & (df["ae_flag"] == 1)).to_numpy()] = "ae_only"
    evidence[(alert & both).to_numpy()] = "both"
    evidence[(alert & ~either).to_numpy()] = "weighted_combination"
    df["model_evidence"] = evidence
    df["detector"] = np.where(alert, "if_ae_ensemble", "none")

    ordered = [
        COL_HOUSEHOLD,
        "timestamp",
        "consumption",
        "anomaly_score",
        "is_anomaly",
        "severity",
        "confidence",
        "detector",
        "model_evidence",
        "if_score",
        "ae_score",
        "if_flag",
        "ae_flag",
        "weight_if",
        "weight_ae",
        "decision_threshold",
    ]
    df = df[ordered].sort_values([COL_HOUSEHOLD, "timestamp"]).reset_index(drop=True)

    log_print(f"Total readings: {len(df):,}")
    log_print(
        f"Ensemble anomalies: {int(df['is_anomaly'].sum()):,} "
        f"({df['is_anomaly'].mean() * 100:.2f}%)"
    )
    log_print("Severity breakdown:")
    log_print(df["severity"].value_counts().to_string())
    log_print("Confidence breakdown:")
    log_print(df["confidence"].value_counts().to_string())
    return df


def save_api_outputs(df: pd.DataFrame) -> None:
    df.to_csv(API_CSV, index=False)

    anomalies = df[df["is_anomaly"] == 1].copy()
    anomalies["timestamp"] = anomalies["timestamp"].astype(str)

    payload = {
        "meta": {
            "decision_model": "validation_calibrated_if_ae_ensemble",
            "total_readings": int(len(df)),
            "total_anomalies": int(len(anomalies)),
            "households": sorted(int(h) for h in df[COL_HOUSEHOLD].unique()),
            "severity_breakdown": {
                str(k): int(v) for k, v in df["severity"].value_counts().items()
            },
            "confidence_breakdown": {
                str(k): int(v) for k, v in df["confidence"].value_counts().items()
            },
            "generated_at": pd.Timestamp.now().isoformat(),
        },
        "anomalies": [],
    }

    for _, row in anomalies.iterrows():
        payload["anomalies"].append(
            {
                "household_id": int(row[COL_HOUSEHOLD]),
                "timestamp": str(row["timestamp"]),
                "consumption": round(float(row["consumption"]), 4),
                "anomaly_score": round(float(row["anomaly_score"]), 6),
                "is_anomaly": int(row["is_anomaly"]),
                "severity": str(row["severity"]),
                "confidence": str(row["confidence"]),
                "detector": str(row["detector"]),
                "model_evidence": str(row["model_evidence"]),
                "if_score": round(float(row["if_score"]), 6),
                "ae_score": round(float(row["ae_score"]), 6),
                "if_flag": int(row["if_flag"]),
                "ae_flag": int(row["ae_flag"]),
                "weight_if": round(float(row["weight_if"]), 4),
                "weight_ae": round(float(row["weight_ae"]), 4),
                "decision_threshold": round(float(row["decision_threshold"]), 6),
            }
        )

    with open(API_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    log_print(f"Saved API CSV: {API_CSV}")
    log_print(f"Saved anomaly JSON: {API_JSON}")


def verify_us14(df: pd.DataFrame) -> bool:
    required = [
        COL_HOUSEHOLD,
        "timestamp",
        "consumption",
        "anomaly_score",
        "is_anomaly",
        "severity",
        "confidence",
        "detector",
        "model_evidence",
        "if_score",
        "ae_score",
        "weight_if",
        "weight_ae",
        "decision_threshold",
    ]
    checks = [(name, name in df.columns) for name in required]
    checks += [
        ("JSON saved", os.path.exists(API_JSON)),
        ("CSV saved", os.path.exists(API_CSV)),
    ]

    ok = all(value for _, value in checks)
    for name, passed in checks:
        log_print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    log_print(f"US-14 status: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


if __name__ == "__main__":
    frame = build_api_payload()
    save_api_outputs(frame)
    verify_us14(frame)
