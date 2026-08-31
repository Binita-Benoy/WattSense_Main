"""
api/tariff.py

Endpoints:
  GET /api/households/{household_id}/tariff  - US-20: view tariff structure
  GET /api/households/{household_id}/cost    - US-21: current cycle cost

Mounted into main.py alongside the existing routes:
  from api import tariff
  app.include_router(tariff.router)
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models import Tariff
from app_services.current_cost import calculate_current_cost

router = APIRouter(prefix="/api/households", tags=["tariff"])


@router.get("/{household_id}/tariff")
def get_tariff(household_id: int, db: Session = Depends(get_db)):
    """
    US-20 acceptance criteria:
      - Show peak rate, off-peak rate, and their time periods.
      - No tariff assigned -> friendly message, not blank/error data.
      - Reflects the latest tariff automatically (query runs fresh
        every request, no caching).
    """
    tariff = (
        db.query(Tariff)
        .filter(Tariff.household_id == household_id)
        .order_by(Tariff.created_at.desc())
        .first()
    )
    if tariff is None:
        return {
            "found": False,
            "message": "No tariff has been assigned to this household yet.",
        }

    return {
        "found": True,
        "tariff": {
            "household_id": tariff.household_id,
            "peak_rate": tariff.peak_rate,
            "off_peak_rate": tariff.off_peak_rate,
            "currency": tariff.currency,
            "peak_period": {
                "days": "Everyday",
                "start": tariff.peak_start_time.strftime("%I:%M %p").lstrip("0"),
                "end": tariff.peak_end_time.strftime("%I:%M %p").lstrip("0"),
            },
            "off_peak_period": {
                "days": "Everyday",
                "description": "All hours outside the peak period",
                "start": tariff.peak_end_time.strftime("%I:%M %p").lstrip("0"),
                "end": tariff.peak_start_time.strftime("%I:%M %p").lstrip("0"),
            },
        },
    }


@router.get("/{household_id}/cost")
def get_current_cost(household_id: int, db: Session = Depends(get_db)):
    result = calculate_current_cost(db, household_id)
    if not result.found and result.message and "No tariff" in result.message:
        raise HTTPException(status_code=404, detail=result.message)
    return result.as_dict()