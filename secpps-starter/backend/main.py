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

# >>> ADDED: needed for the new token-scoped /api/my-household/* routes below
from models_auth import User
from api.auth import get_current_user
# <<< END ADDED

app = FastAPI(title="SECPPS API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#engine = create_engine(DATABASE_URL, pool_pre_ping=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/db")
def health_db():
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"status": "ok"}


# --- Team: mount your routers below as you build them ---------------------
# from api.routes import router as api_router
# app.include_router(api_router, prefix="/api")
from api.auth import router as auth_router
app.include_router(auth_router, prefix="/auth", tags=["auth"])

# >>> ADDED: new routers for US-17 (forecast), US-18 (anomalies), US-19 (reports)
from api.anomaly import router as anomaly_router
app.include_router(anomaly_router, prefix="/api", tags=["anomaly"])

from api.forecast import router as forecast_router
app.include_router(forecast_router, prefix="/api", tags=["forecast"])

from api.reports import router as reports_router
app.include_router(reports_router, prefix="/api", tags=["reports"])
# <<< END ADDED
# >>> ADDED: onboarding choice + new-user forecast (CSV upload)
from api.profile import router as profile_router
app.include_router(profile_router, prefix="/api", tags=["profile"])

from api.new_user_forecast import router as new_user_forecast_router
app.include_router(new_user_forecast_router, prefix="/api", tags=["new-user forecast"])
# <<< END ADDED
# >>> ADDED: reports for a new user's uploaded data (US-19)
from api.new_user_reports import router as new_user_reports_router
app.include_router(new_user_reports_router, prefix="/api", tags=["new-user reports"])
# >>> ADDED: tariff, current cost, predicted cost and recommendations (US-20 to US-24)
from api.tariff import router as tariff_router
app.include_router(tariff_router, prefix="/api", tags=["tariff & cost"])
# <<< END ADDED
# <<< END ADDED

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


# >>> ADDED: token-scoped versions of the two endpoints above.
# SECURITY NOTE: the two old endpoints above (/api/households/{household_id}/...)
# accept household_id from the URL, so anyone with a valid token (or even
# without one, since they're not auth-protected at all) can view ANY
# household's data by changing the number in the URL. These new endpoints
# below fix that for the main dashboard by taking household_id ONLY from
# the logged-in user's own record (current_user.household_id) — never from
# the client. The old endpoints are left in place, unchanged, in case
# anything else (e.g. HouseholdSelector.jsx) still relies on them; just
# don't use them for anything that should be private to one user.
@app.get("/api/my-household/readings")
def get_my_readings(
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    limit: int = Query(2000, le=20000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    household_id = current_user.household_id

    q = db.query(EnergyReading).filter(EnergyReading.household_id == household_id)
    if start:
        q = q.filter(EnergyReading.reading_timestamp >= start)
    if end:
        q = q.filter(EnergyReading.reading_timestamp <= end)
    q = q.order_by(EnergyReading.reading_timestamp).limit(limit)

    rows = q.all()
    if not rows:
        raise HTTPException(status_code=404, detail="No readings found for your household")

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


@app.get("/api/my-household/summary")
def get_my_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    household_id = current_user.household_id

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
        raise HTTPException(status_code=404, detail="No readings found for your household")

    return {
        "household_id": household_id,
        "reading_count": count,
        "avg_kwh": round(avg_kwh, 4) if avg_kwh is not None else None,
        "max_kwh": round(max_kwh, 4) if max_kwh is not None else None,
        "total_kwh": round(total_kwh, 2) if total_kwh is not None else None,
        "first_reading": first_ts.isoformat() if first_ts else None,
        "last_reading": last_ts.isoformat() if last_ts else None,
    }
# <<< END ADDED