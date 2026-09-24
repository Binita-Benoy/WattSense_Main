"""
SECPPS backend — basic working API over the real database.

Endpoints:
  GET /health              - liveness check
  GET /health/db           - confirms DB connectivity
  GET /api/households      - list households (for the dropdown)
  GET /api/households/{household_id}/readings  - time series for charting
  GET /api/households/{household_id}/summary   - quick stats for the dashboard
"""
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from database import get_db, engine
from models import Household, EnergyReading

from app_services.billing import predict_future_cost, StaticForecastProvider, TariffRates
from app_services.recommendation import generate_recommendations
from models import Tariff 

app = FastAPI(title="SECPPS API", version="0.1.0")

# Allow the Vite dev server to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/db")
def health_db():
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.get("/api/households")
def list_households(limit: int = 50, db: Session = Depends(get_db)):
    """Returns households for the dashboard's household picker."""
    rows = db.query(Household).limit(limit).all()
    return [
        {
            "household_id": r.household_id,
            "source": r.source,
            "dwelling_type": r.dwelling_type,
            "climate_zone": r.climate_zone,
            "electricity_use_group": r.electricity_use_group,
            "has_aircon": r.has_aircon,
            "has_generation": r.has_generation,
        }
        for r in rows
    ]


@app.get("/api/households/{household_id}/readings")
def get_readings(
    household_id: int,
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    limit: int = Query(2000, le=20000),
    db: Session = Depends(get_db),
):
    """Returns a time series of consumption for one household, for charting."""
    q = db.query(EnergyReading).filter(EnergyReading.household_id == household_id)
    if start:
        q = q.filter(EnergyReading.reading_timestamp >= start)
    if end:
        q = q.filter(EnergyReading.reading_timestamp <= end)
    q = q.order_by(EnergyReading.reading_timestamp).limit(limit)

    rows = q.all()
    if not rows:
        raise HTTPException(status_code=404, detail="No readings found for this household")

    return [
        {
            "reading_timestamp": r.reading_timestamp.isoformat(),
            "consumption_kwh": r.consumption_kwh,
            "controlled_load_kwh": r.controlled_load_kwh,
            "gross_generation_kwh": r.gross_generation_kwh,
            "net_generation_kwh": r.net_generation_kwh,
        }
        for r in rows
    ]


@app.get("/api/households/{household_id}/summary")
def get_summary(household_id: int, db: Session = Depends(get_db)):
    """Quick aggregate stats for the dashboard's stat cards."""
    result = (
        db.query(
            func.count(EnergyReading.id),
            func.avg(EnergyReading.consumption_kwh),
            func.max(EnergyReading.consumption_kwh),
            func.sum(EnergyReading.consumption_kwh),
            func.min(EnergyReading.reading_timestamp),
            func.max(EnergyReading.reading_timestamp),
        )
        .filter(EnergyReading.household_id == household_id)
        .one()
    )
    count, avg_kwh, max_kwh, total_kwh, first_ts, last_ts = result
    if count == 0:
        raise HTTPException(status_code=404, detail="No readings found for this household")

    return {
        "household_id": household_id,
        "reading_count": count,
        "avg_kwh": round(avg_kwh, 4) if avg_kwh is not None else None,
        "max_kwh": round(max_kwh, 4) if max_kwh is not None else None,
        "total_kwh": round(total_kwh, 2) if total_kwh is not None else None,
        "first_reading": first_ts.isoformat() if first_ts else None,
        "last_reading": last_ts.isoformat() if last_ts else None,
    }

@app.get("/api/households/{household_id}/predicted-cost")
def get_predicted_cost(household_id: int, horizon_days: int = 7, db: Session = Depends(get_db)):
    """US-22: predicted cost for the next `horizon_days` (default 7)."""
    tariff_row = db.query(Tariff).filter(Tariff.household_id == household_id).first()
    if tariff_row is None:
        raise HTTPException(status_code=404, detail="No tariff assigned to this household")
 
    # TODO: swap StaticForecastProvider for Binita's real forecast module once it's ready.
    forecast_provider = StaticForecastProvider(readings=[])
 
    predicted = predict_future_cost(
        tariff_row=tariff_row,
        household_id=household_id,
        forecast_provider=forecast_provider,
        horizon_days=horizon_days,
    )
    return predicted


@app.get("/api/households/{household_id}/recommendations")
def get_recommendations(household_id: int, horizon_days: int = 7, db: Session = Depends(get_db)):
    """US-23: rule-based recommendations built from the predicted cost."""
    tariff_row = db.query(Tariff).filter(Tariff.household_id == household_id).first()
    if tariff_row is None:
        raise HTTPException(status_code=404, detail="No tariff assigned to this household")
 
    forecast_provider = StaticForecastProvider(readings=[])  # TODO: real forecast provider
 
    predicted = predict_future_cost(
        tariff_row=tariff_row,
        household_id=household_id,
        forecast_provider=forecast_provider,
        horizon_days=horizon_days,
    )
 
    tariff = TariffRates(peak_rate=tariff_row.peak_rate, offpeak_rate=tariff_row.offpeak_rate)
 
    # Optional: prior completed cycle's actual cost, from current_cost.py, to enable
    # the "rising cost trend" rule. Leave as None if you don't have that wired up yet.
    prior_cycle_cost = None
 
    recommendations = generate_recommendations(predicted, tariff, prior_cycle_cost)
    return {"predicted_cost": predicted, "recommendations": recommendations}
 