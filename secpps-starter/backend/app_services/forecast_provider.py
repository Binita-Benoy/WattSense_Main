"""
app_services/forecast_provider.py  (US-22)

Turns "what we know about a user's future usage" into hourly ForecastReading
points that billing.calculate_predicted_cost() can price.

Two sources:

  1. The CNN-LSTM forecast (Binita's live_pipeline.py). New users get one when
     they upload their CSV; api/new_user_forecast.py stores the pipeline's
     output in forecast_jobs.result_json. That is used whenever it exists.

  2. A baseline for accounts with no model forecast (dataset households, whose
     /forecast endpoint is still a random mock by team decision, so it is NOT
     used for money). Each future hour is the household's own average for that
     hour of the day (weekday / weekend) over its last LOOKBACK_DAYS of
     readings. Real data, but a "typical week", not a model. The API says
     which source was used.

Why daily totals are split into hours: the pipeline's 7-day and 1-month
horizons are kWh PER DAY, but a tariff charges by time of day. Each day's
total is spread over its 24 hours using the user's own hourly usage shape.
The day's total is kept exactly; only its peak / off-peak split is estimated.
The first 24 hours use the pipeline's real hourly values.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .billing import ForecastReading

LOOKBACK_DAYS = 56  # 8 weeks

Reading = Tuple[datetime, float]
Profile = Dict[Tuple[bool, int], float]  # (is_weekend, hour of day) -> average kWh


def parse_ts(value) -> datetime:
    """ISO timestamp -> naive datetime (readings in the database are naive too)."""
    return datetime.fromisoformat(str(value).strip().replace("Z", "+00:00")).replace(tzinfo=None)


def hourly_profile(readings: List[Reading]) -> Profile:
    """Average kWh for each (weekend?, hour) over the last LOOKBACK_DAYS of readings."""
    if not readings:
        return {}
    cutoff = readings[-1][0] - timedelta(days=LOOKBACK_DAYS)
    buckets: Dict[Tuple[bool, int], List[float]] = {}
    for ts, kwh in readings:
        if ts > cutoff:
            buckets.setdefault((ts.weekday() >= 5, ts.hour), []).append(kwh)
    return {key: sum(values) / len(values) for key, values in buckets.items()}


def baseline_forecast(profile: Profile, start: datetime, hours: int) -> List[ForecastReading]:
    """One point per hour from `start`; an hour never seen in the history is left out, not guessed."""
    out = []
    for h in range(hours):
        ts = start + timedelta(hours=h)
        kwh = profile.get((ts.weekday() >= 5, ts.hour))
        if kwh is not None:
            out.append(ForecastReading(timestamp=ts, predicted_kwh=kwh))
    return out


def _spread_day(day_start: datetime, total_kwh: float, profile: Profile) -> List[ForecastReading]:
    """Split one 24-hour block's total into hourly points using the user's usage shape."""
    hours = [day_start + timedelta(hours=h) for h in range(24)]
    weights = [profile.get((ts.weekday() >= 5, ts.hour), 0.0) for ts in hours]
    weight_sum = sum(weights)
    if weight_sum <= 0:  # no usable history: spread evenly
        weights, weight_sum = [1.0] * 24, 24.0
    return [
        ForecastReading(timestamp=ts, predicted_kwh=total_kwh * w / weight_sum)
        for ts, w in zip(hours, weights)
    ]


def model_forecast(result: dict, profile: Profile, hours: int) -> Tuple[Optional[datetime], List[ForecastReading]]:
    """
    Hourly points for the next `hours` from live_pipeline's result dict:
        result["horizons"]["24h"]["predictions"]    [{"timestamp", "predicted_kwh"}]  hourly
        result["horizons"]["7day"]["predictions"]   [{"day", "date", "predicted_kwh"}] daily totals
        result["horizons"]["1month"]["predictions"] same, 30 days
    Returns (start, points); (None, []) if the result holds nothing usable.
    """
    horizons = (result or {}).get("horizons") or {}

    def predictions(name: str) -> list:
        return (horizons.get(name) or {}).get("predictions") or []  # an {"error": ...} horizon has none

    hourly = [
        ForecastReading(timestamp=parse_ts(p["timestamp"]), predicted_kwh=float(p["predicted_kwh"] or 0.0))
        for p in predictions("24h")
    ]
    # Prefer the 7-day horizon (driven by a real weather forecast) when it covers the window.
    week, month = predictions("7day"), predictions("1month")
    daily = week if (len(week) * 24 >= hours or not month) else month

    if hourly:
        start = hourly[0].timestamp
    elif daily:
        start = parse_ts(daily[0]["date"])
    else:
        return None, []

    # The pipeline's daily blocks are consecutive 24-hour blocks from the first forecast hour.
    by_hour: Dict[datetime, ForecastReading] = {}
    for i, p in enumerate(daily):
        for point in _spread_day(start + timedelta(days=i), float(p["predicted_kwh"] or 0.0), profile):
            by_hour[point.timestamp] = point
    for point in hourly:  # real hourly predictions win over a spread daily total
        by_hour[point.timestamp] = point

    end = start + timedelta(hours=hours)
    return start, [by_hour[ts] for ts in sorted(by_hour) if start <= ts < end]