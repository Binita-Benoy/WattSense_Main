"""
Anomaly routes (US-18), connected to the team's real anomaly detector (US-14).

  GET /api/my-household/anomalies                 check the logged-in user's own data
  GET /api/my-household/anomalies?refresh=true    same, ignoring the short-lived cache

Whose data: ALWAYS the logged-in user's. The household is never taken from the request.
  * an account registered as a NEW user  -> the CSV they uploaded (copied into the database under their own
    household id, see app_services/upload_household.py)
  * any other account (an existing user) -> the dataset household assigned to them

The detector scores the household's whole stored history with the saved models. A check can take a while
(the first one also loads the models), so the answer is cached in memory for a few minutes.
"""
import json
import threading
import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.auth import get_current_user
from app_services import anomaly_inference
from app_services.upload_household import sync_upload_household
from database import get_db
from models_auth import User
from models_profile import ForecastJob, UserProfile

router = APIRouter()

CACHE_SECONDS = 10 * 60
_cache = {}  # key -> (time stored, result)
_cache_lock = threading.Lock()


def _resolve_target(db: Session, user: User):
    """Returns (household_id, source, fingerprint) for whoever is asking."""
    profile = db.query(UserProfile).filter(UserProfile.user_id == user.id).first()
    if profile is not None and profile.user_type == "new":
        job = db.query(ForecastJob).filter(ForecastJob.user_id == user.id).first()
        if job is None or not job.history_json:
            raise HTTPException(
                status_code=404,
                detail="Upload your electricity usage first; there is nothing to check yet.",
            )
        history = json.loads(job.history_json)
        if not history:
            raise HTTPException(status_code=404, detail="Your uploaded file has no readings to check.")
        try:
            household_id = sync_upload_household(db, user.id, history)
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Your uploaded readings couldn't be prepared for the anomaly check ({type(exc).__name__}).",
            ) from exc
        fingerprint = (len(history), history[-1]["timestamp"], round(sum(p["kwh"] for p in history), 3))
        return household_id, "upload", fingerprint
    return user.household_id, "household", None


@router.get("/my-household/anomalies")
def get_my_anomalies(
    refresh: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    household_id, source, fingerprint = _resolve_target(db, current_user)
    key = (household_id, fingerprint)

    if not refresh:
        with _cache_lock:
            hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_SECONDS:
            return hit[1]

    try:
        raw = anomaly_inference.detect_for_household(household_id)
    except anomaly_inference.AnomalyUnavailable as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)

    result = anomaly_inference.shape_response(raw, household_id, source)
    with _cache_lock:
        _cache[key] = (time.time(), result)
    return result