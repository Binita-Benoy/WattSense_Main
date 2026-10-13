"""
app_services/tariffs.py  (US-20)

The tariff structure every cost calculation shares.

The peak window is STANDARD for all households. It is defined once, in
billing.py (PEAK_START / PEAK_END), and re-used here so actual cost, predicted
cost and recommendations can never disagree about what "peak" means. Only the
two rates are stored per household (the `tariffs` table, Tariff in models.py).
"""
from datetime import time
from typing import Optional

from sqlalchemy.orm import Session

from models import Tariff
from .billing import PEAK_END, PEAK_START, is_peak  # noqa: F401  (is_peak re-exported)

CURRENCY = "$"
MAX_RATE = 5.0  # $/kWh. Far above any real retail rate; catches cents typed as dollars.


class TariffError(ValueError):
    """The submitted tariff can't be accepted. The message is safe to show the user."""


def clock_label(t: time) -> str:
    """time(15, 0) -> '3:00 PM'"""
    return t.strftime("%I:%M %p").lstrip("0")


def short_clock(t: time) -> str:
    """time(15, 0) -> '3pm' (used inside recommendation sentences)"""
    return t.strftime("%I:%M%p").lstrip("0").lower().replace(":00", "")


def get_latest_tariff(db: Session, household_id: int) -> Optional[Tariff]:
    """The household's newest tariff row, or None if it has none."""
    return (
        db.query(Tariff)
        .filter(Tariff.household_id == household_id)
        .order_by(Tariff.id.desc())
        .first()
    )


def validate_rates(peak_rate: float, offpeak_rate: float) -> None:
    for name, value in (("Peak rate", peak_rate), ("Off-peak rate", offpeak_rate)):
        if value is None or value != value:  # None or NaN
            raise TariffError(f"{name} is required.")
        if value <= 0:
            raise TariffError(f"{name} must be greater than {CURRENCY}0.")
        if value > MAX_RATE:
            raise TariffError(
                f"{name} looks too high. Enter it in dollars per kWh "
                f"(for example 0.45 for 45 cents), up to {CURRENCY}{MAX_RATE:.2f}."
            )
    if peak_rate < offpeak_rate:
        raise TariffError("Peak rate can't be lower than the off-peak rate.")


def save_tariff(db: Session, household_id: int, peak_rate: float, offpeak_rate: float) -> Tariff:
    """
    Store the rates the user provided. A new row is inserted rather than the
    old one updated, so the history is kept; get_latest_tariff() reads the newest.
    """
    validate_rates(peak_rate, offpeak_rate)
    tariff = Tariff(
        household_id=household_id,
        peak_rate=round(float(peak_rate), 4),
        offpeak_rate=round(float(offpeak_rate), 4),
    )
    db.add(tariff)
    db.commit()
    db.refresh(tariff)
    return tariff


def describe_tariff(tariff: Tariff) -> dict:
    """Peak rate, off-peak rate, peak period, off-peak period."""
    return {
        "peak_rate": tariff.peak_rate,
        "off_peak_rate": tariff.offpeak_rate,
        "currency": CURRENCY,
        "peak_period": {
            "days": "Every day",
            "start": clock_label(PEAK_START),
            "end": clock_label(PEAK_END),
        },
        "off_peak_period": {
            "days": "Every day",
            "start": clock_label(PEAK_END),
            "end": clock_label(PEAK_START),
        },
    }