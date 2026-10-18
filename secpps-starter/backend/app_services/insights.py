
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from models import EnergyReading
from models_profile import ForecastJob, UserProfile
from .billing import ForecastReading, PredictedCost, TariffRates, calculate_predicted_cost
from .current_cost import Reading, calculate_current_cost, split_cost
from .forecast_provider import baseline_forecast, hourly_profile, model_forecast, parse_ts
from .recommendation import generate_recommendations
from .tariffs import CURRENCY, get_latest_tariff, is_peak

RECENT_DAYS = 100  # readings loaded per request: covers the cycle, the 8-week profile and the comparison window
HORIZON_DAYS = {"24h": 1, "7day": 7, "1month": 30}  # same horizon names as the forecast endpoints
MIN_COMPARISON_COVERAGE = 0.9  # only compare against a past period that is (nearly) fully recorded


class NoTariff(Exception):
    """The user hasn't entered a tariff yet, so no cost can be calculated."""


class NoUsage(Exception):
    """There are no readings for this user. The message is safe to show them."""


@dataclass
class Usage:
    source: str                      # "household" (dataset) | "upload" (new user's CSV)
    readings: List[Reading]          # hourly (timestamp, kWh), oldest first, last RECENT_DAYS days
    forecast_result: Optional[dict]  # live_pipeline output, when the user has one


def load_usage(db: Session, user) -> Usage:
    profile = db.query(UserProfile).filter(UserProfile.user_id == user.id).first()
    if profile is not None and profile.user_type == "new":
        job = db.query(ForecastJob).filter(ForecastJob.user_id == user.id).first()
        history = json.loads(job.history_json) if job is not None and job.history_json else []
        if not history:
            raise NoUsage("Upload your electricity usage first; there is nothing to cost yet.")
        readings = sorted((parse_ts(p["timestamp"]), float(p["kwh"] or 0.0)) for p in history)
        cutoff = readings[-1][0] - timedelta(days=RECENT_DAYS)
        result = json.loads(job.result_json) if job.status == "done" and job.result_json else None
        return Usage("upload", [r for r in readings if r[0] > cutoff], result)

    latest = (
        db.query(func.max(EnergyReading.reading_timestamp))
        .filter(EnergyReading.household_id == user.household_id)
        .scalar()
    )
    if latest is None:
        raise NoUsage("No readings found for your household.")
    rows = (
        db.query(EnergyReading.reading_timestamp, EnergyReading.consumption_kwh)
        .filter(
            EnergyReading.household_id == user.household_id,
            EnergyReading.reading_timestamp > latest - timedelta(days=RECENT_DAYS),
            EnergyReading.consumption_kwh.isnot(None),
        )
        .order_by(EnergyReading.reading_timestamp)
        .all()
    )
    return Usage("household", [(ts, kwh) for ts, kwh in rows], None)


def _require_tariff(db: Session, user):
    tariff_row = get_latest_tariff(db, user.household_id)
    if tariff_row is None:
        raise NoTariff("No tariff has been set for your household yet.")
    return tariff_row


# ---------------------------------------------------------------------------
# US-21
# ---------------------------------------------------------------------------
def current_cost(db: Session, user) -> dict:
    tariff_row = _require_tariff(db, user)
    usage = load_usage(db, user)
    data = calculate_current_cost(usage.readings, tariff_row).as_dict()
    data["source"] = usage.source
    return data


# ---------------------------------------------------------------------------
# US-22
# ---------------------------------------------------------------------------
def _predict(db: Session, user, horizon: str):
    tariff_row = _require_tariff(db, user)
    tariff = TariffRates(peak_rate=tariff_row.peak_rate, offpeak_rate=tariff_row.offpeak_rate)
    usage = load_usage(db, user)
    days = HORIZON_DAYS[horizon]
    profile = hourly_profile(usage.readings)

    start, points = model_forecast(usage.forecast_result, profile, days * 24) if usage.forecast_result else (None, [])
    if points:
        forecast = {
            "source": "cnn_lstm",
            "label": "CNN-LSTM model, trained on your data",
            "generated_at": usage.forecast_result.get("generated_at"),
        }
    else:
        # Start where the user's readings stop (the hour after the last one).
        start = usage.readings[-1][0].replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        points = baseline_forecast(profile, start, days * 24)
        forecast = {
            "source": "historical_profile",
            "label": "Baseline: your average usage for each hour of the day, last 8 weeks",
            "generated_at": None,
        }

    predicted = calculate_predicted_cost(
        household_id=user.household_id,
        period_start=start,
        period_end=start + timedelta(days=days),
        readings=points,
        tariff=tariff,
        interval_hours=1.0,
    )
    return tariff_row, tariff, usage, points, predicted, forecast


def _predicted_dict(predicted: PredictedCost, tariff: TariffRates, forecast: dict, horizon: str) -> dict:
    data = asdict(predicted)
    data.pop("household_id", None)
    data["period_start"] = predicted.period_start.isoformat()
    data["period_end"] = predicted.period_end.isoformat()
    data["predicted_total_kwh"] = round(predicted.predicted_total_kwh, 3)
    data["horizon"] = horizon
    data["horizon_days"] = HORIZON_DAYS[horizon]
    data["currency"] = CURRENCY
    data["peak_rate"] = tariff.peak_rate
    data["off_peak_rate"] = tariff.offpeak_rate
    data["forecast"] = forecast
    return data


def _breakdown(points: List[ForecastReading], tariff: TariffRates, hourly: bool) -> List[dict]:
    """Predicted kWh and cost per hour (24h horizon) or per calendar day, for the chart."""
    buckets: "OrderedDict[str, dict]" = OrderedDict()
    for p in sorted(points, key=lambda p: p.timestamp):
        label = p.timestamp.isoformat() if hourly else p.timestamp.date().isoformat()
        bucket = buckets.setdefault(label, {"label": label, "peak_kwh": 0.0, "offpeak_kwh": 0.0})
        bucket["peak_kwh" if is_peak(p.timestamp) else "offpeak_kwh"] += p.predicted_kwh
    out = []
    for b in buckets.values():
        peak_cost = b["peak_kwh"] * tariff.peak_rate
        offpeak_cost = b["offpeak_kwh"] * tariff.offpeak_rate
        out.append({
            "label": b["label"],
            "peak_kwh": round(b["peak_kwh"], 3),
            "offpeak_kwh": round(b["offpeak_kwh"], 3),
            "peak_cost": round(peak_cost, 2),
            "offpeak_cost": round(offpeak_cost, 2),
            "total_cost": round(peak_cost + offpeak_cost, 2),
        })
    return out


def predicted_cost(db: Session, user, horizon: str = "7day") -> dict:
    _, tariff, _, points, predicted, forecast = _predict(db, user, horizon)
    data = _predicted_dict(predicted, tariff, forecast, horizon)
    data["breakdown_unit"] = "hour" if horizon == "24h" else "day"
    data["breakdown"] = _breakdown(points, tariff, hourly=horizon == "24h")
    return data


# ---------------------------------------------------------------------------
# US-23 / US-24
# ---------------------------------------------------------------------------
def recommendations(db: Session, user, horizon: str = "7day") -> dict:
    tariff_row, tariff, usage, _, predicted, forecast = _predict(db, user, horizon)
    days = HORIZON_DAYS[horizon]

    # What the same length of time actually cost just before the forecast starts.
    comparison_cost: Optional[float] = None
    window_start = predicted.period_start - timedelta(days=days)
    before = [(ts, kwh) for ts, kwh in usage.readings if window_start <= ts < predicted.period_start]
    if len(before) >= days * 24 * MIN_COMPARISON_COVERAGE:
        comparison_cost = split_cost(before, tariff_row)["total_cost"]

    label = "the 24 hours before" if days == 1 else f"the {days} days before"
    recs = generate_recommendations(predicted, tariff, comparison_cost, comparison_label=label)
    total_saving = round(sum(r.estimated_saving or 0.0 for r in recs), 2)

    return {
        "predicted_cost": _predicted_dict(predicted, tariff, forecast, horizon),
        "previous_period_cost": comparison_cost,
        "estimated_total_saving": total_saving,
        "recommended_cost": round(max(predicted.predicted_total_cost - total_saving, 0.0), 2),
        "recommendations": [r.as_dict() for r in recs],
    }