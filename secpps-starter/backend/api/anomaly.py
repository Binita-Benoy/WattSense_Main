"""
Anomaly routes (US-18).

IMPORTANT — current data source is MOCK:
The real anomaly detection pipeline (ml_pipeline/anomaly_detection/) only
covers 5 specific households from its own training run, and its output
currently lives outside this backend's Docker build context entirely
(ml_pipeline/ sits outside backend/). Per team decision, we're using a
COPY of that real output (backend/data/anomaly_mock.json) as placeholder
data to get the dashboard working now.

TO SWAP IN THE REAL PIPELINE LATER:
Replace the body of _load_anomalies() below with whatever the real
integration turns out to be (e.g. a live endpoint, or a DB table Alka
loads the results into) — the route and response shape don't need to
change, since they already match the real pipeline's actual field names.
"""

import json
import os
from fastapi import APIRouter, Depends

from models_auth import User
from api.auth import get_current_user

router = APIRouter()

_MOCK_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "anomaly_mock.json")


def _load_anomalies():
    with open(_MOCK_PATH) as f:
        return json.load(f)


@router.get("/my-household/anomalies")
def get_my_anomalies(current_user: User = Depends(get_current_user)):
    """
    Returns anomalies for the logged-in user's own household.

    Since the mock data only covers 5 specific households, most users will
    get an empty list with a friendly message rather than an error — this
    is expected right now, not a bug.
    """
    data = _load_anomalies()
    household_id = current_user.household_id

    matching = [
        a for a in data["anomalies"] if a["household_id"] == household_id
    ]

    return {
        "household_id": household_id,
        "anomaly_count": len(matching),
        "anomalies": matching,
        "message": (
            None
            if matching
            else "No anomaly data is available for your household yet."
        ),
    }