
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd

from config import (
    AE_MODEL_PATH,
    AE_PREPROCESS_PKL,
    AE_SCALER_PKL,
    AE_THRESHOLD_PKL,
    API_CSV,
    API_JSON,
    API_MAX_ANOMALIES_PER_HOUSEHOLD,
    API_RECENT_HOURS,
    API_REQUIRE_HOURLY_HISTORY,
    COL_HOUSEHOLD,
    COL_KWH,
    COL_TIMESTAMP,
    ENSEMBLE_CONFIG_PKL,
    ENSEMBLE_RESULTS_CSV,
    IF_MODEL_PKL,
    IF_PREPROCESS_PKL,
    IF_SCALER_PKL,
    IF_THRESHOLD_PKL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
)
from db_loader import (
    get_engine,
    get_inference_household_ranges,
    list_inference_household_ids,
    load_inference_household_histories,
)
from feature_engineering.us10_features import build_feature_matrix
from logging_utils import log_print
from anomaly_detection.ensemble_detector import (
    load_ensemble_config,
    predict_ensemble_from_results,
)
from anomaly_detection.us11_isolation_forest import predict_if_on_split
from anomaly_detection.us12_autoencoder import predict_dataframe

from forecasting.ensure_weather import ensure_historical_weather

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceArtifacts:
    """Trained/calibrated objects required for live anomaly inference."""

    if_trained: dict
    ae_model: Any
    ae_seq_data: dict
    ensemble_config: dict


def _require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} not found: {path}. Run run_anomaly_detection.py first."
        )


def load_inference_artifacts() -> InferenceArtifacts:
    """Load trained model artefacts once when the backend process starts."""
    required = [
        (IF_MODEL_PKL, "Isolation Forest model"),
        (IF_SCALER_PKL, "Isolation Forest scaler"),
        (IF_PREPROCESS_PKL, "Isolation Forest preprocessing"),
        (IF_THRESHOLD_PKL, "Isolation Forest threshold"),
        (AE_MODEL_PATH, "LSTM Autoencoder model"),
        (AE_SCALER_PKL, "LSTM Autoencoder scaler"),
        (AE_PREPROCESS_PKL, "LSTM Autoencoder preprocessing"),
        (AE_THRESHOLD_PKL, "LSTM Autoencoder threshold"),
        (ENSEMBLE_CONFIG_PKL, "Ensemble configuration"),
    ]
    for path, label in required:
        _require_file(path, label)

    if_preprocess = joblib.load(IF_PREPROCESS_PKL)
    if_trained = {
        **if_preprocess,
        "model": joblib.load(IF_MODEL_PKL),
        "scaler": joblib.load(IF_SCALER_PKL),
        "threshold": float(joblib.load(IF_THRESHOLD_PKL)),
    }

    try:
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for US-14 because the saved LSTM "
            "Autoencoder participates in anomaly inference."
        ) from exc

    ae_preprocess = joblib.load(AE_PREPROCESS_PKL)
    ae_seq_data = {
        **ae_preprocess,
        "scaler": joblib.load(AE_SCALER_PKL),
        "threshold": float(joblib.load(AE_THRESHOLD_PKL)),
    }

    return InferenceArtifacts(
        if_trained=if_trained,
        ae_model=load_model(AE_MODEL_PATH, compile=False),
        ae_seq_data=ae_seq_data,
        ensemble_config=load_ensemble_config(),
    )


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------
def _coerce_household_ids(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, (int, np.integer, str)):
        values = [values]

    ids: list[int] = []
    for value in values:
        try:
            ids.append(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid household_id: {value!r}") from exc
    return sorted(set(ids))


def resolve_api_households(payload: Any) -> tuple[list[int], bool, str | None]:
    """Resolve household IDs from a backend request.

    Returns ``(ids, all_households, source_filter)``.
    """
    if isinstance(payload, (int, np.integer, str, list, tuple, set)):
        ids = _coerce_household_ids(payload)
        if not ids:
            raise ValueError("At least one household_id is required.")
        return ids, False, None

    if not isinstance(payload, Mapping):
        raise ValueError(
            "API output component expects household_id, household_ids, or all_households=true."
        )

    all_households = bool(payload.get("all_households", False))
    source = payload.get("source")
    source = str(source) if source not in (None, "") else None

    if all_households:
        ids = list_inference_household_ids(source=source)
        if not ids:
            label = f" with source={source!r}" if source else ""
            raise ValueError(f"No households with energy readings were found{label}.")
        return ids, True, source

    if "household_ids" in payload:
        ids = _coerce_household_ids(payload.get("household_ids"))
    else:
        ids = _coerce_household_ids(payload.get("household_id"))

    if not ids:
        raise ValueError(
            "Provide 'household_id', 'household_ids', or set 'all_households' to true."
        )
    return ids, False, source


# ---------------------------------------------------------------------------
# Input data checks
# ---------------------------------------------------------------------------
def _history_quality(raw_history: pd.DataFrame, strict_hourly: bool) -> list[dict[str, Any]]:
    """Validate one-row-per-hour cadence and describe the loaded DB history."""
    if raw_history.empty:
        return []

    data = raw_history.reset_index().copy()
    data[COL_TIMESTAMP] = pd.to_datetime(data[COL_TIMESTAMP])
    quality: list[dict[str, Any]] = []
    problems: list[str] = []

    for hh_id, group in data.groupby(COL_HOUSEHOLD, sort=True):
        ts = group[COL_TIMESTAMP].sort_values().drop_duplicates().reset_index(drop=True)
        diffs = ts.diff().dropna()
        irregular = diffs[diffs != pd.Timedelta(hours=1)]

        quality.append(
            {
                "household_id": int(hh_id),
                "received_readings": int(len(ts)),
                "history_start": ts.iloc[0].isoformat(),
                "history_end": ts.iloc[-1].isoformat(),
                "irregular_intervals": int(len(irregular)),
                "hourly_cadence": bool(len(irregular) == 0),
            }
        )

        if len(irregular):
            problems.append(
                f"HH {int(hh_id)} has {len(irregular)} non-hourly interval(s)."
            )

    if strict_hourly and problems:
        raise ValueError(
            "US-10 lag features assume one reading per hour. " + " ".join(problems)
        )
    return quality


def _weather_coverage(raw_history: pd.DataFrame) -> dict[int, dict[str, Any]]:
    """Return simple weather coverage diagnostics for API metadata."""
    if raw_history.empty:
        return {}

    result: dict[int, dict[str, Any]] = {}
    for hh_id, group in raw_history.groupby(COL_HOUSEHOLD, sort=True):
        total = int(len(group))
        row = {"rows": total}
        for col in ("temperature", "humidity", "shortwave_radiation"):
            present = int(group[col].notna().sum()) if col in group.columns else 0
            row[f"{col}_rows"] = present
            row[f"{col}_coverage"] = round(present / total, 4) if total else 0.0
        result[int(hh_id)] = row
    return result


# ---------------------------------------------------------------------------
# Frontend formatting
# ---------------------------------------------------------------------------
def assign_severity(score: float, is_anomaly: int) -> str:
    if int(is_anomaly) == 0 or not np.isfinite(float(score)):
        return "NORMAL"
    if float(score) >= SEVERITY_HIGH:
        return "HIGH"
    if float(score) >= SEVERITY_MEDIUM:
        return "MEDIUM"
    return "LOW"


def _format_prediction_frame(ensemble_results: pd.DataFrame) -> pd.DataFrame:
    if ensemble_results.empty:
        return pd.DataFrame()

    df = ensemble_results.copy()
    if "timestamp" not in df.columns:
        df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["anomaly_score"] = pd.to_numeric(
        df["ensemble_score"], errors="coerce"
    ).clip(0, 1)
    df["is_anomaly"] = pd.to_numeric(
        df["is_anomaly"], errors="coerce"
    ).fillna(0).astype("int8")
    df["if_flag"] = pd.to_numeric(
        df.get("if_flag", 0), errors="coerce"
    ).fillna(0).astype("int8")
    df["ae_flag"] = pd.to_numeric(
        df.get("ae_flag", 0), errors="coerce"
    ).fillna(0).astype("int8")
    df["if_score"] = pd.to_numeric(df.get("if_score"), errors="coerce")
    df["ae_score"] = pd.to_numeric(df.get("ae_score"), errors="coerce")

    df["severity"] = [
        assign_severity(score, flag)
        for score, flag in zip(df["anomaly_score"], df["is_anomaly"])
    ]

    alert = df["is_anomaly"] == 1
    both = (df["if_flag"] == 1) & (df["ae_flag"] == 1)
    either = (df["if_flag"] == 1) | (df["ae_flag"] == 1)
    df["model_agreement"] = np.select(
        [alert & both, alert & either, alert],
        ["HIGH", "MEDIUM", "LOW"],
        default="NORMAL",
    )
    # Compatibility with the earlier frontend contract.  This means base-model
    # agreement, not a calibrated statistical probability.
    df["confidence"] = df["model_agreement"]

    evidence = np.full(len(df), "none", dtype=object)
    evidence[(alert & (df["if_flag"] == 1) & (df["ae_flag"] == 0)).to_numpy()] = "if_only"
    evidence[(alert & (df["if_flag"] == 0) & (df["ae_flag"] == 1)).to_numpy()] = "ae_only"
    evidence[(alert & both).to_numpy()] = "both"
    evidence[(alert & ~either).to_numpy()] = "weighted_combination"
    df["model_evidence"] = evidence
    df["detector"] = "validation_calibrated_if_ae_ensemble"

    ordered = [
        COL_HOUSEHOLD,
        "timestamp",
        "consumption",
        "anomaly_score",
        "is_anomaly",
        "severity",
        "model_agreement",
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
    for col in ordered:
        if col not in df.columns:
            df[col] = np.nan

    return df[ordered].sort_values(
        [COL_HOUSEHOLD, "timestamp"]
    ).reset_index(drop=True)


def _record(row: pd.Series) -> dict[str, Any]:
    return {
        "household_id": int(row[COL_HOUSEHOLD]),
        "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
        "consumption": round(float(row["consumption"]), 4),
        "anomaly_score": round(float(row["anomaly_score"]), 6),
        "is_anomaly": int(row["is_anomaly"]),
        "severity": str(row["severity"]),
        "model_agreement": str(row["model_agreement"]),
        "confidence": str(row["confidence"]),
        "detector": str(row["detector"]),
        "model_evidence": str(row["model_evidence"]),
        "if_score": round(float(row["if_score"]), 6) if pd.notna(row["if_score"]) else None,
        "ae_score": round(float(row["ae_score"]), 6) if pd.notna(row["ae_score"]) else None,
        "if_flag": int(row["if_flag"]),
        "ae_flag": int(row["ae_flag"]),
        "weight_if": round(float(row["weight_if"]), 4),
        "weight_ae": round(float(row["weight_ae"]), 4),
        "decision_threshold": round(float(row["decision_threshold"]), 6),
    }


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [_record(row) for _, row in df.iterrows()]


def _latest_records(formatted: pd.DataFrame) -> list[dict[str, Any]]:
    if formatted.empty:
        return []
    latest = (
        formatted.sort_values([COL_HOUSEHOLD, "timestamp"])
        .groupby(COL_HOUSEHOLD, as_index=False, sort=True)
        .tail(1)
    )
    return _records(latest)


def build_db_api_response(
    formatted: pd.DataFrame,
    *,
    quality: list[dict[str, Any]],
    feature_rows_by_household: dict[int, int],
    weather_status: dict[int, dict[str, Any]],
    weather_coverage: dict[int, dict[str, Any]],
    include_normal: bool = False,
    recent_hours: int = API_RECENT_HOURS,
    max_anomalies_per_household: int = API_MAX_ANOMALIES_PER_HOUSEHOLD,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build bounded JSON-friendly output for the frontend."""
    warnings = warnings or []
    quality_by_hh = {int(x["household_id"]): x for x in quality}

    summaries: list[dict[str, Any]] = []
    recent_parts: list[pd.DataFrame] = []
    anomaly_parts: list[pd.DataFrame] = []

    if formatted.empty:
        return {
            "meta": {
                "decision_model": "validation_calibrated_if_ae_ensemble",
                "received_readings": int(sum(x["received_readings"] for x in quality)),
                "scored_readings": 0,
                "total_anomalies": 0,
                "households": sorted(quality_by_hh),
                "score_semantics": "0=more normal, 1=more anomalous; not a probability",
                "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            },
            "summary_by_household": [],
            "latest_by_household": [],
            "recent_readings": [],
            "readings": [],
            "anomalies": [],
            "weather_status": weather_status,
            "weather_coverage": weather_coverage,
            "warnings": warnings,
        }

    for hh_id, group in formatted.groupby(COL_HOUSEHOLD, sort=True):
        hh = int(hh_id)
        group = group.sort_values("timestamp")
        anom = group[group["is_anomaly"] == 1]
        q = quality_by_hh.get(hh, {})
        received = int(q.get("received_readings", len(group)))
        feature_rows = int(feature_rows_by_household.get(hh, 0))
        scored = int(len(group))
        latest = group.iloc[-1]

        summaries.append(
            {
                "household_id": hh,
                "history_start": q.get("history_start"),
                "history_end": q.get("history_end"),
                "readings_received": received,
                "feature_rows": feature_rows,
                "readings_scored": scored,
                "warmup_or_unscored": max(received - scored, 0),
                "anomalies": int(len(anom)),
                "anomaly_rate": round(float(len(anom) / scored), 6) if scored else 0.0,
                "high_severity": int((group["severity"] == "HIGH").sum()),
                "medium_severity": int((group["severity"] == "MEDIUM").sum()),
                "max_anomaly_score": round(float(group["anomaly_score"].max()), 6),
                "latest_is_anomaly": int(latest["is_anomaly"]),
                "latest_anomaly_score": round(float(latest["anomaly_score"]), 6),
                "latest_severity": str(latest["severity"]),
                "weather": weather_status.get(hh, {}),
            }
        )

        if recent_hours > 0:
            recent_parts.append(group.tail(int(recent_hours)))
        if max_anomalies_per_household > 0:
            anomaly_parts.append(anom.tail(int(max_anomalies_per_household)))

    first = formatted.iloc[0]
    recent = pd.concat(recent_parts, ignore_index=True) if recent_parts else formatted.iloc[0:0]
    anomalies_returned = (
        pd.concat(anomaly_parts, ignore_index=True)
        if anomaly_parts else formatted.iloc[0:0]
    )
    all_anomalies = formatted[formatted["is_anomaly"] == 1]

    return {
        "meta": {
            "decision_model": "validation_calibrated_if_ae_ensemble",
            "received_readings": int(sum(x["received_readings"] for x in quality)),
            "feature_rows": int(sum(feature_rows_by_household.values())),
            "scored_readings": int(len(formatted)),
            "total_anomalies": int(len(all_anomalies)),
            "returned_anomalies": int(len(anomalies_returned)),
            "households": sorted(int(h) for h in formatted[COL_HOUSEHOLD].unique()),
            "weight_if": round(float(first["weight_if"]), 4),
            "weight_ae": round(float(first["weight_ae"]), 4),
            "decision_threshold": round(float(first["decision_threshold"]), 6),
            "score_semantics": "0=more normal, 1=more anomalous; not a probability",
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        },
        "summary_by_household": summaries,
        "latest_by_household": _latest_records(formatted),
        "recent_readings": _records(recent),
        "readings": _records(formatted) if include_normal else [],
        "anomalies": _records(anomalies_returned),
        "weather_status": weather_status,
        "weather_coverage": weather_coverage,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Inference service
# ---------------------------------------------------------------------------
class AnomalyInferenceService:
    """Reusable backend service using already-trained anomaly models."""

    def __init__(self, artifacts: InferenceArtifacts | None = None) -> None:
        self.artifacts = artifacts or load_inference_artifacts()

    def _ensure_weather_for_households(
        self,
        household_ids: Iterable[int],
        *,
        required: bool = False,
    ) -> tuple[dict[int, dict[str, Any]], list[str]]:
        """Call weather helper before model feature generation.

        If ``required=False`` (default), an Open-Meteo/network failure is
        reported as a warning and inference continues.  Missing weather model
        inputs are then handled by the training-only medians.  Set ``required=True`` if the API should fail instead.
        """
        engine = get_engine()
        status: dict[int, dict[str, Any]] = {}
        warnings: list[str] = []

        for hh_id in household_ids:
            hh = int(hh_id)
            try:
                ensure_historical_weather(engine, household_id=hh)
                status[hh] = {"ensured": True, "error": None}
            except Exception as exc:
                message = f"HH {hh}: historical weather could not be ensured: {exc}"
                status[hh] = {"ensured": False, "error": str(exc)}
                if required:
                    raise RuntimeError(message) from exc
                warnings.append(message)
                logger.warning(message)

        return status, warnings

    def detect_households(
        self,
        household_ids: Iterable[int],
        *,
        include_normal: bool = False,
        ensure_weather: bool = True,
        weather_required: bool = False,
        strict_hourly: bool = API_REQUIRE_HOURLY_HISTORY,
        recent_hours: int = API_RECENT_HOURS,
        max_anomalies_per_household: int = API_MAX_ANOMALIES_PER_HOUSEHOLD,
    ) -> dict[str, Any]:
        """Detect anomalies for one or many households already stored in DB."""
        ids = _coerce_household_ids(household_ids)
        if not ids:
            raise ValueError("At least one household_id is required.")

        # Fail early with a clear message if consumption has not been inserted.
        ranges = get_inference_household_ranges(ids)
        found = set(int(v) for v in ranges[COL_HOUSEHOLD].tolist()) if not ranges.empty else set()
        missing = [h for h in ids if h not in found]
        if missing:
            raise ValueError(
                "No consumption history exists in energy_readings for household(s): "
                + ", ".join(str(h) for h in missing)
                + ". Insert the household history before calling US-14."
            )

        warnings: list[str] = []
        weather_status = {h: {"ensured": False, "skipped": True, "error": None} for h in ids}
        if ensure_weather:
            weather_status, weather_warnings = self._ensure_weather_for_households(
                ids,
                required=weather_required,
            )
            warnings.extend(weather_warnings)

        # Load AFTER weather ensuring so newly inserted Open-Meteo rows are part
        # of the same inference request.
        raw_history = load_inference_household_histories(ids)
        if raw_history.empty:
            raise ValueError("No usable household history could be loaded from PostgreSQL.")

        quality = _history_quality(raw_history, strict_hourly=strict_hourly)
        coverage = _weather_coverage(raw_history)

        # Build exactly the same causal feature definitions used for training.
        # save=False prevents live inference from overwriting feature_matrix.csv.
        features = build_feature_matrix(raw_history, save=False)
        feature_rows_by_household = {
            int(hh): int(len(group))
            for hh, group in features.groupby(COL_HOUSEHOLD)
        }

        # saved/calibrated Isolation Forest.
        if_results = predict_if_on_split(
            features,
            self.artifacts.if_trained,
            split_name="api_db_history",
        )

        # saved LSTM Autoencoder.  Since the complete engineered history
        # is supplied as F_target, the first sequence requires a short warm-up.
        empty_history = features.iloc[0:0].copy()
        ae_results = predict_dataframe(
            self.artifacts.ae_model,
            features,
            empty_history,
            self.artifacts.ae_seq_data,
            split_name="api_db_history",
        )

        # validation-locked IF+AE weights/decision threshold.
        ensemble = predict_ensemble_from_results(
            if_results,
            ae_results,
            self.artifacts.ensemble_config,
            split_name="api_db_history",
        )

        # Require scores only from detectors that actually have non-zero saved
        # ensemble weight.  This matters with the current calibration where IF
        # may have zero weight and AE may have full weight.
        w_if = float(self.artifacts.ensemble_config.get("weight_if", 0.0))
        w_ae = float(self.artifacts.ensemble_config.get("weight_ae", 0.0))
        active = np.ones(len(ensemble), dtype=bool)
        if w_if > 1e-12:
            active &= ensemble["if_score"].notna().to_numpy()
        if w_ae > 1e-12:
            active &= ensemble["ae_score"].notna().to_numpy()
        ensemble = ensemble.loc[active].copy()

        formatted = _format_prediction_frame(ensemble)
        if formatted.empty:
            warnings.append(
                "No rows could be scored after feature/sequence warm-up. "
                "Provide a longer hourly consumption history."
            )

        return build_db_api_response(
            formatted,
            quality=quality,
            feature_rows_by_household=feature_rows_by_household,
            weather_status=weather_status,
            weather_coverage=coverage,
            include_normal=include_normal,
            recent_hours=recent_hours,
            max_anomalies_per_household=max_anomalies_per_household,
            warnings=warnings,
        )

    def detect_request(self, payload: Any) -> dict[str, Any]:
        """Backend-friendly entry point for household_id(s)/all_households."""
        ids, _all_households, _source = resolve_api_households(payload)

        options = payload if isinstance(payload, Mapping) else {}
        return self.detect_households(
            ids,
            include_normal=bool(options.get("include_normal", False)),
            ensure_weather=bool(options.get("ensure_weather", True)),
            weather_required=bool(options.get("weather_required", False)),
            strict_hourly=bool(options.get("strict_hourly", API_REQUIRE_HOURLY_HISTORY)),
            recent_hours=int(options.get("recent_hours", API_RECENT_HOURS)),
            max_anomalies_per_household=int(
                options.get(
                    "max_anomalies_per_household",
                    API_MAX_ANOMALIES_PER_HOUSEHOLD,
                )
            ),
        )

    # Backward-friendly name for API route code.
    def detect(self, payload: Any) -> dict[str, Any]:
        return self.detect_request(payload)


_SERVICE: AnomalyInferenceService | None = None


def get_inference_service() -> AnomalyInferenceService:
    """Singleton so TensorFlow/model artefacts load once per backend process."""
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = AnomalyInferenceService()
    return _SERVICE


def detect_anomalies(payload: Any) -> dict[str, Any]:
    """Convenience function for a backend POST endpoint."""
    return get_inference_service().detect_request(payload)


# ---------------------------------------------------------------------------
# Offline export retained for the existing run_anomaly_detection.py workflow
# ---------------------------------------------------------------------------
def build_api_payload(ensemble_results: pd.DataFrame | None = None) -> pd.DataFrame:
    """Format already-generated ensemble test results for legacy CSV/JSON export."""
    log_print("\n" + "=" * 70)
    log_print("US-14: BUILDING FRONTEND/API PAYLOAD")
    log_print("=" * 70)

    if ensemble_results is None:
        if not os.path.exists(ENSEMBLE_RESULTS_CSV):
            raise FileNotFoundError(
                f"Ensemble results not found: {ENSEMBLE_RESULTS_CSV}. "
                "Run run_anomaly_detection.py first."
            )
        ensemble_results = pd.read_csv(
            ENSEMBLE_RESULTS_CSV,
            parse_dates=["timestamp"],
        )

    df = _format_prediction_frame(ensemble_results)
    if df.empty:
        raise ValueError("Ensemble results are empty.")
    return df


def save_api_outputs(df: pd.DataFrame) -> None:
    """Preserve the project's existing offline CSV/JSON output files."""
    df.to_csv(API_CSV, index=False)
    anomalies = df[df["is_anomaly"] == 1]
    first = df.iloc[0]
    payload = {
        "meta": {
            "decision_model": "validation_calibrated_if_ae_ensemble",
            "scored_readings": int(len(df)),
            "total_anomalies": int(len(anomalies)),
            "weight_if": round(float(first["weight_if"]), 4),
            "weight_ae": round(float(first["weight_ae"]), 4),
            "decision_threshold": round(float(first["decision_threshold"]), 6),
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        },
        "anomalies": _records(anomalies),
    }
    with open(API_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    log_print(f"Saved API CSV: {API_CSV}")
    log_print(f"Saved API JSON: {API_JSON}")


def verify_us14(df: pd.DataFrame) -> bool:
    required = [
        COL_HOUSEHOLD,
        "timestamp",
        "consumption",
        "anomaly_score",
        "is_anomaly",
        "severity",
        "model_agreement",
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
