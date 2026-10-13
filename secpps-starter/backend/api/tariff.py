"""
Tariff, cost and recommendation routes (US-20 to US-24).

  GET /api/my-household/tariff           US-20  the tariff structure (rates + periods)
  PUT /api/my-household/tariff           US-20  provide / change the two rates
  GET /api/my-household/cost             US-21  cost of recorded usage this billing cycle
  GET /api/my-household/cost/predicted   US-22  cost of forecast usage (?horizon=24h|7day|1month)
  GET /api/my-household/advice           US-23/24  recommendations (?horizon=...)

Whose data: ALWAYS the logged-in user's (see app_services/insights.py). Nothing
is taken from the request except the rates and the horizon.

Mounted in main.py like the other routers:
    from api.tariff import router as tariff_router
    app.include_router(tariff_router, prefix="/api", tags=["tariff & cost"])
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.auth import get_current_user
from app_services import insights
from app_services.tariffs import TariffError, describe_tariff, get_latest_tariff, save_tariff
from database import engine, get_db
from models import Tariff
from models_auth import User

# The dump has no `tariffs` table. Create it (from the Tariff model in models.py)
# if it is missing; existing tables and data are never touched.
try:
    Tariff.__table__.create(bind=engine, checkfirst=True)
except Exception:
    pass  # database not restored yet; the table is created on the next start after the restore

router = APIRouter()

HORIZON = Query("7day", pattern="^(24h|7day|1month)$")


class TariffIn(BaseModel):
    peak_rate: float       # $ per kWh
    off_peak_rate: float   # $ per kWh


def _run(fn, *args):
    """Turn the two expected 'can't calculate yet' cases into 404s the page can explain."""
    try:
        return fn(*args)
    except (insights.NoTariff, insights.NoUsage) as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/my-household/tariff")
def get_my_tariff(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    tariff = get_latest_tariff(db, current_user.household_id)
    if tariff is None:
        return {"found": False, "message": "No tariff has been set for your household yet."}
    return {"found": True, "tariff": describe_tariff(tariff)}


@router.put("/my-household/tariff")
def put_my_tariff(
    body: TariffIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        tariff = save_tariff(db, current_user.household_id, body.peak_rate, body.off_peak_rate)
    except TariffError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"found": True, "tariff": describe_tariff(tariff)}


@router.get("/my-household/cost")
def get_my_cost(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return _run(insights.current_cost, db, current_user)


@router.get("/my-household/cost/predicted")
def get_my_predicted_cost(
    horizon: str = HORIZON,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return _run(insights.predicted_cost, db, current_user, horizon)


@router.get("/my-household/advice")
def get_my_advice(
    horizon: str = HORIZON,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Each recommendation carries: message, explanation (why), expected_benefit, priority."""
    return _run(insights.recommendations, db, current_user, horizon)