
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from .tariffs import CURRENCY, is_peak

Reading = Tuple[datetime, float]  # (timestamp, kWh used in that hour)


def _cycle_start(moment: datetime) -> datetime:
    return datetime(moment.year, moment.month, 1)


def split_cost(readings: List[Reading], tariff) -> dict:
    """Sum readings into peak / off-peak kWh and cost them at the tariff's two rates."""
    peak_kwh = sum(kwh for ts, kwh in readings if is_peak(ts))
    off_peak_kwh = sum(kwh for ts, kwh in readings if not is_peak(ts))
    peak_cost = round(peak_kwh * tariff.peak_rate, 2)
    off_peak_cost = round(off_peak_kwh * tariff.offpeak_rate, 2)
    return {
        "peak_kwh": round(peak_kwh, 2),
        "off_peak_kwh": round(off_peak_kwh, 2),
        "peak_cost": peak_cost,
        "off_peak_cost": off_peak_cost,
        "total_cost": round(peak_cost + off_peak_cost, 2),
    }


@dataclass
class CostResult:
    total_cost: float
    peak_cost: float
    off_peak_cost: float
    peak_kwh: float
    off_peak_kwh: float
    cycle_start: str
    cycle_end: str
    is_historical: bool
    message: str
    currency: str = CURRENCY

    def as_dict(self) -> dict:
        return asdict(self)


def calculate_current_cost(readings: List[Reading], tariff, now: Optional[datetime] = None) -> CostResult:
    """
    `readings` must be sorted oldest first. Returns a RUNNING total: the cycle
    isn't closed, so it is never presented as a final bill.
    """
    now = now or datetime.utcnow()
    latest = readings[-1][0] if readings else None

    is_historical = latest is not None and latest < _cycle_start(now)
    cycle_end = latest if is_historical else now
    cycle_start = _cycle_start(cycle_end)

    in_cycle = [(ts, kwh) for ts, kwh in readings if cycle_start <= ts <= cycle_end]
    costs = split_cost(in_cycle, tariff)

    if not in_cycle:
        message = "No consumption has been recorded yet for the current billing cycle."
    elif is_historical:
        message = (
            f"Your latest readings are from {latest.strftime('%d %b %Y').lstrip('0')}, "
            "so this is the billing cycle they fall in."
        )
    else:
        message = "Running estimate. It updates as more readings arrive this cycle."

    return CostResult(
        cycle_start=cycle_start.isoformat(),
        cycle_end=cycle_end.isoformat(),
        is_historical=is_historical,
        message=message,
        **costs,
    )