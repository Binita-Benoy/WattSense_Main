
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