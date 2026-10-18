
from datetime import datetime, timedelta
from random import uniform
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database import get_db
from models import EnergyReading
from models_auth import User
from api.auth import get_current_user

router = APIRouter()

HORIZON_HOURS = {"24h": 24, "7day": 168, "1month": 720}


def _generate_mock_forecast(last_timestamp, last_kwh, horizon: str):
    predictions = []
    kwh = last_kwh or 1.2
    for i in range(1, HORIZON_HOURS[horizon] + 1):
        ts = last_timestamp + timedelta(hours=i)
        kwh = max(0.0, kwh + uniform(-0.15, 0.15))
        predictions.append({"timestamp": ts.isoformat(), "predicted_kwh": round(kwh, 4)})
    return predictions


@router.get("/my-household/forecast")
def get_my_forecast(
    horizon: str = Query("24h", pattern="^(24h|7day|1month)$"),
    as_of: Optional[datetime] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    household_id = current_user.household_id

    q = db.query(EnergyReading).filter(EnergyReading.household_id == household_id)
    if as_of:
        q = q.filter(EnergyReading.reading_timestamp <= as_of)
    last_reading = q.order_by(EnergyReading.reading_timestamp.desc()).first()

    if last_reading is None:
        raise HTTPException(
            status_code=404,
            detail="No historical readings found for your household, cannot generate a forecast.",
        )

    return {
        "household_id": household_id,
        "horizon": horizon,
        "unit": "kWh per hour",
        "predictions": _generate_mock_forecast(last_reading.reading_timestamp, last_reading.consumption_kwh, horizon),
        "model": "mock_v0 (placeholder, not the trained model; existing households aren't connected to it yet)",
    }


# >>> ADDED: existing households now get the REAL forecast (Binita's bidirectional CNN-LSTM pipeline).
# The household's recent hourly readings are copied from the database into the same background job that new users
# use (api/new_user_forecast.py -> app_services/forecast_runner.py), and the page reads the result from
# GET /api/forecast/new-user. The mock above is left in place (GET /my-household/forecast is not removed).
import json
import os
import threading
from datetime import timezone

import pandas as pd
from sqlalchemy import func

from fastapi import status
from app_services.consumption_csv import CsvValidationError, normalize_consumption_csv
from api.new_user_forecast import _run_job
from models_profile import ForecastJob

# The dataset has no location for a household, so weather is fetched for one place (all SGSC homes are in NSW).
HOUSEHOLD_LOCATION = os.getenv("HOUSEHOLD_LOCATION", "Newcastle, Australia")
HOUSEHOLD_COUNTRY = os.getenv("HOUSEHOLD_COUNTRY", "AU")
HOUSEHOLD_STATE = os.getenv("HOUSEHOLD_STATE", "NSW")
HOUSEHOLD_FORECAST_DAYS = int(os.getenv("HOUSEHOLD_FORECAST_DAYS", "90"))  # history given to the model


@router.post("/my-household/forecast/run", status_code=status.HTTP_202_ACCEPTED)
def run_my_household_forecast(
    force: bool = Query(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Start (or reuse) the real forecast for the logged-in user's dataset household."""
    existing = db.query(ForecastJob).filter(ForecastJob.user_id == current_user.id).first()
    if existing and existing.status in ("queued", "running"):
        return {"status": existing.status}
    if existing and existing.status == "done" and not force:
        return {"status": "done"}

    household_id = current_user.household_id
    last = (
        db.query(func.max(EnergyReading.reading_timestamp))
        .filter(EnergyReading.household_id == household_id)
        .scalar()
    )
    if last is None:
        raise HTTPException(status_code=404, detail="No historical readings found for your household.")
    rows = (
        db.query(EnergyReading.reading_timestamp, EnergyReading.consumption_kwh)
        .filter(
            EnergyReading.household_id == household_id,
            EnergyReading.reading_timestamp >= last - pd.Timedelta(days=HOUSEHOLD_FORECAST_DAYS),
            EnergyReading.reading_timestamp <= last,
        )
        .order_by(EnergyReading.reading_timestamp)
        .all()
    )
    raw = pd.DataFrame(rows, columns=["timestamp", "kwh"]).to_csv(index=False).encode()
    try:
        df, info, notes = normalize_consumption_csv(raw)  # same checks/clean-up as an uploaded file
    except CsvValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    history = [
        {"timestamp": t.strftime("%Y-%m-%dT%H:%M:%S"), "kwh": float(v)}
        for t, v in zip(df["timestamp"], df["kwh"])
    ]
    csv_text = df.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S")
    fields = dict(
        status="queued", progress="Waiting to start", error=None,
        location=HOUSEHOLD_LOCATION, country_code=HOUSEHOLD_COUNTRY, state=HOUSEHOLD_STATE,
        warnings_json=json.dumps(notes), info_json=json.dumps(info),
        history_json=json.dumps(history), result_json=None,
        created_at=datetime.now(timezone.utc), started_at=None, finished_at=None,
    )
    if existing:
        for key, value in fields.items():
            setattr(existing, key, value)
        job = existing
    else:
        job = ForecastJob(user_id=current_user.id, **fields)
        db.add(job)
    db.commit()
    db.refresh(job)

    threading.Thread(
        target=_run_job,
        args=(job.id, csv_text, HOUSEHOLD_LOCATION, HOUSEHOLD_COUNTRY, HOUSEHOLD_STATE),
        daemon=True,
    ).start()
    return {"status": "queued"}
#