from logging_utils import log_print
import logging
logger = logging.getLogger()
# anomaly_detection/us14_api.py
# ─────────────────────────────────────────────────────────────────────────────
# US-14: Provide anomaly results to the application
#
# Acceptance criteria:
# ✅ household
# ✅ timestamp
# ✅ consumption
# ✅ anomaly score
# ✅ severity/flag
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import json
import os

from config import (
    IF_RESULTS_CSV, AE_RESULTS_CSV,
    API_JSON, API_CSV, COL_HOUSEHOLD,
    SEVERITY_HIGH, SEVERITY_MEDIUM
)


def assign_severity(score: float, is_anomaly: int) -> str:
    """
    Convert score + flag to human-readable severity.

    NORMAL: not flagged as anomaly
    LOW:    flagged but low score
    MEDIUM: flagged, moderate score
    HIGH:   flagged, high score (immediate attention)
    """
    if is_anomaly == 0:
        return "NORMAL"
    if score >= SEVERITY_HIGH:
        return "HIGH"
    if score >= SEVERITY_MEDIUM:
        return "MEDIUM"
    return "LOW"


def build_api_payload() -> pd.DataFrame:
    """
    Merge IF and AE results into the dashboard API format.

    Output columns (all AC satisfied):
        household_id    ← AC: household
        timestamp       ← AC: timestamp
        consumption     ← AC: consumption
        anomaly_score   ← AC: anomaly score
        is_anomaly      ← AC: flag
        severity        ← AC: severity
        confidence      — HIGH if both detectors agree
        detector        — which model(s) flagged
        if_score        — raw IF score
        ae_score        — raw AE score
    """
    log_print("\n" + "="*60)
    log_print("US-14: BUILDING API PAYLOAD")
    log_print("="*60)

    # ── Load results ───────────────────────────────────────────────────────
    df_if = pd.read_csv(IF_RESULTS_CSV,
                        parse_dates=["timestamp"])
    df_ae = pd.read_csv(AE_RESULTS_CSV,
                        parse_dates=["timestamp"])

    # Rename for merge clarity
    df_if = df_if.rename(columns={
        "is_anomaly":    "if_flag",
        "anomaly_score": "if_score"
    })
    df_ae = df_ae.rename(columns={
        "is_anomaly":    "ae_flag",
        "anomaly_score": "ae_score"
    })

    # ── Merge on household + timestamp ─────────────────────────────────────
    key = [COL_HOUSEHOLD, "timestamp"]
    df  = df_if[key + ["consumption", "if_flag", "if_score"]]\
          .merge(
              df_ae[key + ["ae_flag", "ae_score"]],
              on=key, how="left"
          )

    # Fill missing AE values
    df["ae_flag"]  = df["ae_flag"].fillna(0).astype(int)
    df["ae_score"] = df["ae_score"].fillna(0.0)

    # ── Combined anomaly score — average of both ───────────────────────────
    df["anomaly_score"] = (
        (df["if_score"] * 0.5 + df["ae_score"] * 0.5)
        .round(6)
    )

    # ── Combined flag — anomaly if EITHER detector flags ──────────────────
    df["is_anomaly"] = (
        (df["if_flag"] == 1) | (df["ae_flag"] == 1)
    ).astype(int)

    # ── Severity ───────────────────────────────────────────────────────────
    df["severity"] = df.apply(
        lambda r: assign_severity(r["anomaly_score"],
                                  r["is_anomaly"]),
        axis=1
    )

    # ── Confidence — both agree = high ─────────────────────────────────────
    def get_confidence(row) -> str:
        if row["if_flag"] == 1 and row["ae_flag"] == 1:
            return "HIGH"
        if row["if_flag"] == 1 or row["ae_flag"] == 1:
            return "MEDIUM"
        return "NORMAL"

    df["confidence"] = df.apply(get_confidence, axis=1)

    # ── Detector label ─────────────────────────────────────────────────────
    def get_detector(row) -> str:
        det = []
        if row["if_flag"] == 1:
            det.append("isolation_forest")
        if row["ae_flag"] == 1:
            det.append("lstm_autoencoder")
        return ",".join(det) if det else "none"

    df["detector"] = df.apply(get_detector, axis=1)

    # ── Column order ───────────────────────────────────────────────────────
    ordered = [
        COL_HOUSEHOLD,   # AC: household
        "timestamp",     # AC: timestamp
        "consumption",   # AC: consumption
        "anomaly_score", # AC: anomaly score
        "is_anomaly",    # AC: flag
        "severity",      # AC: severity
        "confidence",
        "detector",
        "if_score",
        "ae_score"
    ]
    ordered = [c for c in ordered if c in df.columns]
    df = df[ordered].sort_values([COL_HOUSEHOLD, "timestamp"])

    # ── Summary ────────────────────────────────────────────────────────────
    total = len(df)
    anom  = df["is_anomaly"].sum()
    log_print(f"\n  Total readings: {total:,}")
    log_print(f"  Anomalies:      {anom:,} ({anom/total*100:.1f}%)")
    log_print(f"\n  Severity breakdown:")
    log_print(df["severity"].value_counts().to_string())
    log_print(f"\n  Confidence breakdown:")
    log_print(df["confidence"].value_counts().to_string())

    return df


def save_api_outputs(df: pd.DataFrame) -> None:
    """
    Save API payload as CSV and JSON.

    CSV  → Alka loads into database for dashboard queries
    JSON → Jasim's REST endpoint serves directly to frontend

    JSON contains only anomalous readings to keep
    file size manageable for the dashboard API.
    """

    # ── CSV (all readings) ─────────────────────────────────────────────────
    df.to_csv(API_CSV, index=False)
    log_print(f"\n  ✓ CSV (all readings): {API_CSV}")

    # ── JSON (anomalies only) ──────────────────────────────────────────────
    anomalies = df[df["is_anomaly"] == 1].copy()
    anomalies["timestamp"] = anomalies["timestamp"].astype(str)

    payload = {
        "meta": {
            "total_readings":    int(len(df)),
            "total_anomalies":   int(len(anomalies)),
            "households":        sorted(
                [int(h) for h in df[COL_HOUSEHOLD].unique()]
            ),
            "severity_breakdown": df["severity"]
                                   .value_counts().to_dict(),
            "confidence_breakdown": df["confidence"]
                                    .value_counts().to_dict(),
            "generated_at":      pd.Timestamp.now().isoformat()
        },
        "anomalies": []
    }

    for _, row in anomalies.iterrows():
        payload["anomalies"].append({
            "household_id":  int(row[COL_HOUSEHOLD]),
            "timestamp":     str(row["timestamp"]),
            "consumption":   round(float(row["consumption"]), 4),
            "anomaly_score": round(float(row["anomaly_score"]), 6),
            "is_anomaly":    int(row["is_anomaly"]),
            "severity":      str(row["severity"]),
            "confidence":    str(row["confidence"]),
            "detector":      str(row["detector"]),
            "if_score":      round(float(row.get("if_score", 0)), 6),
            "ae_score":      round(float(row.get("ae_score", 0)), 6)
        })

    with open(API_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    log_print(f"  ✓ JSON (anomalies):   {API_JSON}")
    log_print(f"\n  JSON preview (first 2 anomalies):")
    log_print(json.dumps(payload["anomalies"][:2], indent=2))


def verify_us14(df: pd.DataFrame) -> bool:
    log_print(f"\n{'='*60}")
    log_print("US-14 ACCEPTANCE CRITERIA")
    log_print("="*60)
    criteria = [
        ("household column",
         COL_HOUSEHOLD in df.columns),
        ("timestamp column",
         "timestamp" in df.columns),
        ("consumption column",
         "consumption" in df.columns),
        ("anomaly_score column",
         "anomaly_score" in df.columns),
        ("is_anomaly flag",
         "is_anomaly" in df.columns),
        ("severity column",
         "severity" in df.columns),
        ("JSON saved",
         os.path.exists(API_JSON)),
        ("CSV saved",
         os.path.exists(API_CSV)),
    ]
    all_pass = True
    for name, passed in criteria:
        icon = "✅" if passed else "❌"
        log_print(f"  {icon}  {name}")
        if not passed:
            all_pass = False

    log_print(f"\n  {'ALL PASSED ✅' if all_pass else 'SOME FAILED ❌'}")
    log_print(f"\n  Sample anomaly records:")
    sample_cols = [COL_HOUSEHOLD, "timestamp", "consumption",
                   "anomaly_score", "severity", "confidence"]
    sample_cols = [c for c in sample_cols if c in df.columns]
    log_print(df[df["is_anomaly"]==1][sample_cols]
          .head(3).to_string(index=False))

    return all_pass


if __name__ == "__main__":
    df = build_api_payload()
    save_api_outputs(df)
    verify_us14(df)
