"""
Bridge between the web API and the team's anomaly service (backend/ml/anomaly_detection/us14_api.py).

That service is database-driven: give it a household_id that has hourly readings in `energy_readings`
and it loads the readings (fetching weather first if it is missing), rebuilds the training features,
scores them with the SAVED Isolation Forest + LSTM autoencoder + calibrated ensemble, and returns JSON.
Nothing is retrained here (see backend/ml/US14_DB_WEATHER_INTEGRATION_V6.md).

This module only does the plumbing the service itself doesn't:
  * puts backend/ml on the import path (the ml code imports its siblings by plain name),
  * points the ml code at the same database as the web app (it reads DB_HOST/DB_USER/..., not DATABASE_URL),
  * loads the models once, lazily, on the first request (not at server start-up),
  * runs one check at a time (TensorFlow is memory-hungry),
  * turns failures into messages a person can act on.
"""
import logging
import os
import sys
import threading

from sqlalchemy.engine import make_url

log = logging.getLogger("anomaly")

ML_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml"))
MODELS_DIR = os.path.join(ML_DIR, "outputs", "models")

# The service's own defaults, copied here so the web layer can pass them explicitly.
RECENT_HOURS = 7 * 24
MAX_ANOMALIES = 500

_service = None
_load_lock = threading.Lock()
_run_lock = threading.Lock()


class AnomalyUnavailable(Exception):
    """Raised with a message that is safe to show to the user, and the HTTP status to use."""

    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _prepare_environment() -> None:
    if ML_DIR not in sys.path:
        sys.path.append(ML_DIR)  # appended, so nothing in backend/ can be shadowed by an ml module
    url = os.getenv("DATABASE_URL")
    if url:
        parts = make_url(url)
        for key, value in (("DB_HOST", parts.host), ("DB_PORT", parts.port), ("DB_NAME", parts.database),
                           ("DB_USER", parts.username), ("DB_PASSWORD", parts.password)):
            if value is not None:
                os.environ.setdefault(key, str(value))  # an explicit DB_* setting still wins


def get_service():
    """Load the saved models once; later calls reuse them."""
    global _service
    if _service is not None:
        return _service
    with _load_lock:
        if _service is None:
            _prepare_environment()
            try:
                from anomaly_detection.us14_api import get_inference_service

                _service = get_inference_service()
            except FileNotFoundError as exc:
                raise AnomalyUnavailable(
                    "The anomaly detection models haven't been saved yet, so nothing can be checked. "
                    f"{exc} Copy the teammates' backend/ml/outputs/ folder into place, or train them "
                    "with run_anomaly_detection.py."
                ) from exc
            except ImportError as exc:
                raise AnomalyUnavailable(f"A library the anomaly detector needs is missing: {exc}") from exc
    return _service


def detect_for_household(household_id: int) -> dict:
    """Run the saved models over one stored household and return the service's raw response."""
    service = get_service()
    request = {
        "household_id": int(household_id),
        "include_normal": False,
        "ensure_weather": True,
        "weather_required": False,   # if the weather service is unreachable, carry on and say so
        "recent_hours": RECENT_HOURS,
        "max_anomalies_per_household": MAX_ANOMALIES,
    }
    with _run_lock:
        try:
            return service.detect_request(request)
        except ValueError as exc:      # e.g. readings aren't hourly, or no history stored
            raise AnomalyUnavailable(str(exc), 400) from exc
        except RuntimeError as exc:
            raise AnomalyUnavailable(str(exc), 502) from exc
        except Exception as exc:       # noqa: BLE001
            name = type(exc).__name__
            if "OperationalError" in name or "psycopg2" in type(exc).__module__:
                raise AnomalyUnavailable("The anomaly check couldn't reach the database.", 503) from exc
            log.exception("Anomaly check failed for household %s", household_id)
            raise AnomalyUnavailable(f"The anomaly check failed unexpectedly ({name}).", 500) from exc


def _clean(value):
    """NaN/inf -> None so the JSON is valid for the browser."""
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def shape_response(raw: dict, household_id: int, source: str) -> dict:
    """Turn the service's multi-household response into one household's result for the dashboard."""
    hh = int(household_id)

    def mine(rows):
        return [r for r in rows if int(r.get("household_id", -1)) == hh]

    summary = next((s for s in raw.get("summary_by_household", []) if int(s["household_id"]) == hh), None)
    latest = next(iter(mine(raw.get("latest_by_household", []))), None)
    anomalies = [
        {k: a[k] for k in ("timestamp", "consumption", "anomaly_score", "severity", "model_agreement",
                           "model_evidence", "detector")}
        for a in mine(raw.get("anomalies", []))
    ]
    recent = [
        {k: r[k] for k in ("timestamp", "consumption", "anomaly_score", "is_anomaly", "severity")}
        for r in mine(raw.get("recent_readings", []))
    ]
    meta = raw.get("meta", {})
    weather = (raw.get("weather_status") or {}).get(hh) or (raw.get("weather_status") or {}).get(str(hh)) or {}

    result = {
        "household_id": hh,
        "source": source,                       # "household" (dataset) or "upload" (the user's own file)
        "status": "ok" if summary else "no_scored_readings",
        "anomaly_count": int(summary["anomalies"]) if summary else 0,
        "returned_anomalies": len(anomalies),
        "summary": None if not summary else {
            k: summary[k] for k in ("history_start", "history_end", "readings_received", "readings_scored",
                                    "anomaly_rate", "high_severity", "medium_severity", "max_anomaly_score")
        },
        "latest": None if not latest else {
            k: latest[k] for k in ("timestamp", "consumption", "anomaly_score", "is_anomaly", "severity")
        },
        "anomalies": anomalies,
        "recent_readings": recent,
        "warnings": list(raw.get("warnings", [])),
        "weather": weather,
        "meta": {
            "decision_model": meta.get("decision_model"),
            "decision_threshold": meta.get("decision_threshold"),
            "score_semantics": meta.get("score_semantics"),
            "generated_at": meta.get("generated_at"),
        },
        "message": None if summary else "There isn't enough history to score yet. A longer hourly history is needed.",
    }
    return _clean(result)