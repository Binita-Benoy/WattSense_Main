"""
Forecast routes (US-17) for dataset households ("existing users").

IMPORTANT: this is still a MOCK, by team decision. The trained model
(ml/forecasting/cnn/) is built for brand-new households (live_pipeline.py: it needs an
uploaded CSV + location). There is no function yet that forecasts an already-registered
household from the data in our database. New users get the real model through
api/new_user_forecast.py. Replace _generate_mock_forecast() when that gap is closed.

CHANGED vs the first version: optional `as_of`, so the forecast continues from the last reading
at or before the end of the date range shown on the chart (no gap), and the unit is "kWh per hour"
for every horizon (the mock always produces hourly points; it used to say "per day" for two of them).
"""
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