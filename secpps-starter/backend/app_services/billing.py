

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Iterable, List, Optional, Protocol

# ---------------------------------------------------------------------------
# Tariff windows — kept identical to current_cost.py so actual and predicted
# costs are always computed on the same rules.
# ---------------------------------------------------------------------------
PEAK_START = time(15, 0)   # 3pm
PEAK_END = time(21, 0)     # 9pm


def is_peak(ts: datetime) -> bool:
    """Classify a timestamp as peak (3pm-9pm) or off-peak (all other times)."""
    return PEAK_START <= ts.time() < PEAK_END


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TariffRates:
    """Minimal view of the household's Tariff row (US-20)."""
    peak_rate: float       # $/kWh
    offpeak_rate: float    # $/kWh


@dataclass(frozen=True)
class ForecastReading:
    """One predicted-consumption interval, as produced by forecast module."""
    timestamp: datetime
    predicted_kwh: float


class ForecastProvider(Protocol):
    """Adapter boundary onto forecasting module."""

    def get_forecast(
        self, household_id: int, start: datetime, end: datetime
    ) -> List[ForecastReading]:
        ...


@dataclass(frozen=True)
class PredictedCost:
    """Forecasted cost for a future window."""
    household_id: int
    period_start: datetime
    period_end: datetime
    predicted_peak_kwh: float
    predicted_offpeak_kwh: float
    predicted_peak_cost: float
    predicted_offpeak_cost: float
    predicted_total_cost: float
    is_partial_forecast: bool  # True if forecast data didn't cover the full window
    covered_hours: float
    expected_hours: float

    @property
    def predicted_total_kwh(self) -> float:
        return self.predicted_peak_kwh + self.predicted_offpeak_kwh


# ---------------------------------------------------------------------------
# Pure calculation logic (DB/forecast-source agnostic — unit-testable
# against a plain list of ForecastReading, same approach used to test
# current_cost.py against the local SQLite replica).
# ---------------------------------------------------------------------------
def calculate_predicted_cost(
    household_id: int,
    period_start: datetime,
    period_end: datetime,
    readings: Iterable[ForecastReading],
    tariff: TariffRates,
    interval_hours: float = 0.5,
) -> PredictedCost:
    """
    Sum forecasted consumption into peak/off-peak buckets and cost them at
    the household's agreed tariff rates.

    `interval_hours` is the duration each ForecastReading represents (e.g.
    0.5 for half-hourly forecast points, 1.0 for hourly) — used only to
    flag partial-data coverage, not in the cost arithmetic itself (each
    predicted_kwh is already a total for its own interval).
    """
    peak_kwh = 0.0
    offpeak_kwh = 0.0
    covered_intervals = 0

    for r in readings:
        if not (period_start <= r.timestamp < period_end):
            continue
        covered_intervals += 1
        if is_peak(r.timestamp):
            peak_kwh += r.predicted_kwh
        else:
            offpeak_kwh += r.predicted_kwh

    expected_hours = (period_end - period_start).total_seconds() / 3600
    covered_hours = covered_intervals * interval_hours
    is_partial = covered_hours < expected_hours - 1e-9

    peak_cost = round(peak_kwh * tariff.peak_rate, 2)
    offpeak_cost = round(offpeak_kwh * tariff.offpeak_rate, 2)

    return PredictedCost(
        household_id=household_id,
        period_start=period_start,
        period_end=period_end,
        predicted_peak_kwh=round(peak_kwh, 3),
        predicted_offpeak_kwh=round(offpeak_kwh, 3),
        predicted_peak_cost=peak_cost,
        predicted_offpeak_cost=offpeak_cost,
        predicted_total_cost=round(peak_cost + offpeak_cost, 2),
        is_partial_forecast=is_partial,
        covered_hours=round(covered_hours, 2),
        expected_hours=round(expected_hours, 2),
    )


# ---------------------------------------------------------------------------
# Service entry point — takes the household's Tariff row (however the caller
# queried it — main.py does `db.query(Tariff).filter(...).first()` directly,
# same as get_summary()) and a forecast provider, and returns the prediction.
# No DB session is threaded through here: the endpoint in main.py does the
# lookup and the "no tariff assigned" 404, exactly like it already does for
# "no readings found" in get_readings()/get_summary().
# ---------------------------------------------------------------------------
def predict_future_cost(
    tariff_row,
    household_id: int,
    forecast_provider: ForecastProvider,
    horizon_days: int = 7,
    start: Optional[datetime] = None,
) -> PredictedCost:
    """
    Predict the household's electricity cost for the next `horizon_days`
    (default: next 7 days from now, or from `start` if given).

    `tariff_row` is the SQLAlchemy Tariff row already fetched by the caller
    (e.g. `db.query(Tariff).filter(Tariff.household_id == household_id).first()`
    in main.py) — pass None-checking responsibility (the "no tariff assigned"
    case) to the caller, same as the 404s already in main.py.
    """
    tariff = TariffRates(
        peak_rate=tariff_row.peak_rate,
        offpeak_rate=tariff_row.offpeak_rate,
    )

    period_start = start or datetime.now()
    period_end = period_start + timedelta(days=horizon_days)

    readings = forecast_provider.get_forecast(household_id, period_start, period_end)

    return calculate_predicted_cost(
        household_id=household_id,
        period_start=period_start,
        period_end=period_end,
        readings=readings,
        tariff=tariff,
    )


class StaticForecastProvider:
    """Wraps a pre-built list of readings — for tests/demos only."""

    def __init__(self, readings: List[ForecastReading]):
        self._readings = readings

    def get_forecast(
        self, household_id: int, start: datetime, end: datetime
    ) -> List[ForecastReading]:
        return [r for r in self._readings if start <= r.timestamp < end]