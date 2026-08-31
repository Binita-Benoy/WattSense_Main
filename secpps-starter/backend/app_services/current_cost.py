"""
app_services/current_cost.py

Calculates the cost of ACTUAL recorded consumption so far in the current
billing cycle, using a household's Tariff (peak/off-peak rate + period).

This is deliberately separate from billing.py, which projects cost from
FORECASTED future usage — different inputs, different question. Both
modules should import the shared Tariff model from models.py rather
than each defining their own tariff structure, to avoid divergence.

ASSUMPTION: "current billing cycle" = calendar month (1st 00:00 -> now).
There's no billing_cycles table in the schema yet. If the real billing
cycle definition differs (e.g. a rolling anniversary date), update
_current_billing_cycle_start() below — nothing else needs to change.
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from models import Tariff, EnergyReading


def _is_peak(moment: datetime, tariff: Tariff) -> bool:
    t = moment.time()
    return tariff.peak_start_time <= t < tariff.peak_end_time


def _current_billing_cycle_start(now: Optional[datetime] = None) -> datetime:
    now = now or datetime.utcnow()
    return datetime(now.year, now.month, 1)


@dataclass
class CostResult:
    found: bool
    message: Optional[str] = None
    total_cost: Optional[float] = None
    peak_cost: Optional[float] = None
    off_peak_cost: Optional[float] = None
    peak_kwh: Optional[float] = None
    off_peak_kwh: Optional[float] = None
    currency: str = "$"
    display_text: Optional[str] = None
    cycle_start: Optional[str] = None
    cycle_end: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


def calculate_current_cost(db: Session, household_id: int) -> CostResult:
    """
    US-21 acceptance criteria:
      - Calculate total estimated cost for the current billing cycle
        using consumption (kWh), tariff (peak/off-peak rate), and
        time period (peak/off-peak).
      - Peak consumption charged at peak rate; off-peak at off-peak rate.
      - Sum both into a single total.
      - Missing/no data -> appropriate message, not a false-final total.
      - Running total, since the cycle isn't closed yet.
    """
    tariff = (
        db.query(Tariff)
        .filter(Tariff.household_id == household_id)
        .order_by(Tariff.created_at.desc())
        .first()
    )
    if tariff is None:
        return CostResult(
            found=False,
            message="No tariff has been assigned to this household yet.",
        )

    cycle_start = _current_billing_cycle_start()
    cycle_end = datetime.utcnow()

    readings = (
        db.query(EnergyReading)
        .filter(
            EnergyReading.household_id == household_id,
            EnergyReading.reading_timestamp >= cycle_start,
            EnergyReading.reading_timestamp <= cycle_end,
        )
        .all()
    )

    if not readings:
        return CostResult(
            found=True,
            message="No consumption has been recorded yet for the "
                    "current billing cycle.",
            total_cost=0.0,
            peak_cost=0.0,
            off_peak_cost=0.0,
            peak_kwh=0.0,
            off_peak_kwh=0.0,
            currency=tariff.currency,
            display_text=f"{tariff.currency}0.00 for this billing cycle (so far)",
            cycle_start=cycle_start.isoformat(),
            cycle_end=cycle_end.isoformat(),
        )

    peak_kwh = sum(
        r.consumption_kwh or 0.0 for r in readings
        if _is_peak(r.reading_timestamp, tariff)
    )
    off_peak_kwh = sum(
        r.consumption_kwh or 0.0 for r in readings
        if not _is_peak(r.reading_timestamp, tariff)
    )

    peak_cost = round(peak_kwh * tariff.peak_rate, 2)
    off_peak_cost = round(off_peak_kwh * tariff.off_peak_rate, 2)
    total_cost = round(peak_cost + off_peak_cost, 2)

    return CostResult(
        found=True,
        message="Running estimate; will update as more readings come in "
                "this cycle.",
        total_cost=total_cost,
        peak_cost=peak_cost,
        off_peak_cost=off_peak_cost,
        peak_kwh=round(peak_kwh, 2),
        off_peak_kwh=round(off_peak_kwh, 2),
        currency=tariff.currency,
        display_text=f"{tariff.currency}{total_cost:.2f} for this billing "
                      f"cycle (running estimate)",
        cycle_start=cycle_start.isoformat(),
        cycle_end=cycle_end.isoformat(),
    )