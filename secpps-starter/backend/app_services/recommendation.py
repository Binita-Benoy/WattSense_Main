"""
US-23 - Generate energy-saving recommendations
US-24 - what each recommendation carries for the dashboard
------------------------------------------------------------
Rule-based (not ML). Every recommendation is generated from the household's
real predicted cost (US-22) and its own tariff, and carries the numbers that
triggered it - no generic tips.

Pattern from the story:
    high predicted peak consumption -> recommend shifting usage -> explain why

Each Recommendation has the four things US-24 asks the dashboard to show:
    message           the recommendation
    explanation       why it was generated
    expected_benefit  what following it is expected to achieve
    priority          high / medium / low

This module does not talk to the DB. Callers assemble the inputs and pass
them in, which keeps every rule unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Optional

from .billing import PredictedCost, TariffRates
from .tariffs import PEAK_START, PEAK_END, short_clock


class Severity(str, Enum):
    INFO = "info"        # positive reinforcement / no action needed
    ADVISORY = "advisory"
    WARNING = "warning"  # meaningfully high predicted peak cost


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


_PRIORITY_FOR = {
    Severity.WARNING: Priority.HIGH,
    Severity.ADVISORY: Priority.MEDIUM,
    Severity.INFO: Priority.LOW,
}
_PRIORITY_ORDER = [Priority.HIGH, Priority.MEDIUM, Priority.LOW]


@dataclass(frozen=True)
class Recommendation:
    rule: str                  # which rule fired, for traceability/testing
    severity: Severity
    title: str                 # short heading for the card
    message: str               # the recommendation itself
    explanation: str           # why, grounded in the actual numbers
    expected_benefit: str      # what the household gains by following it
    estimated_saving: Optional[float] = None  # $ over the forecast window, when computable

    @property
    def priority(self) -> Priority:
        return _PRIORITY_FOR[self.severity]

    def as_dict(self) -> dict:
        data = asdict(self)
        data["severity"] = self.severity.value
        data["priority"] = self.priority.value
        return data


# ---------------------------------------------------------------------------
# Thresholds - named constants so they're easy to tune without touching rules.
# ---------------------------------------------------------------------------
HIGH_PEAK_SHARE_THRESHOLD = 0.55   # peak kWh as a share of total predicted kWh
MODERATE_PEAK_SHARE_THRESHOLD = 0.35
LOW_PEAK_SHARE_THRESHOLD = 0.15
MIN_KWH_FOR_ADVICE = 1.0           # ignore near-zero forecasts (not enough signal)
RISING_COST_THRESHOLD = 0.20       # flag a predicted cost >= 20% above the comparison period

# Share of predicted peak usage each rule assumes can realistically be moved
# off-peak, used only for the estimated saving shown with the recommendation.
HIGH_PEAK_SHIFT_SHARE = 0.5
MODERATE_PEAK_SHIFT_SHARE = 0.25
LIGHT_PEAK_SHIFT_SHARE = 0.15

_WINDOW = f"{short_clock(PEAK_START)}-{short_clock(PEAK_END)}"
_OUTSIDE = f"before {short_clock(PEAK_START)} or after {short_clock(PEAK_END)}"


def _peak_share(predicted: PredictedCost) -> float:
    total = predicted.predicted_total_kwh
    if total <= 0:
        return 0.0
    return predicted.predicted_peak_kwh / total


def _rate_gap(tariff: TariffRates) -> float:
    """How much more expensive peak is than off-peak, per kWh."""
    return max(tariff.peak_rate - tariff.offpeak_rate, 0.0)


def _window_label(predicted: PredictedCost) -> str:
    days = round((predicted.period_end - predicted.period_start).total_seconds() / 86400)
    return f"the next {days} days" if days != 1 else "the next 24 hours"


# ---------------------------------------------------------------------------
# Rules - each returns a Recommendation or None.
# ---------------------------------------------------------------------------
def _rule_high_peak_consumption(predicted: PredictedCost, tariff: TariffRates) -> Optional[Recommendation]:
    """High predicted peak consumption -> recommend shifting usage -> explain why."""
    if predicted.predicted_total_kwh < MIN_KWH_FOR_ADVICE:
        return None
    share = _peak_share(predicted)
    if share < HIGH_PEAK_SHARE_THRESHOLD:
        return None

    gap = _rate_gap(tariff)
    shiftable_kwh = predicted.predicted_peak_kwh * HIGH_PEAK_SHIFT_SHARE
    saving = round(shiftable_kwh * gap, 2) if gap > 0 else None

    return Recommendation(
        rule="high_peak_consumption",
        severity=Severity.WARNING,
        title="Move usage out of peak hours",
        message=(
            f"Shift some of your usage out of the {_WINDOW} peak window: run the washing "
            f"machine, dishwasher, dryer or EV charging {_OUTSIDE}."
        ),
        explanation=(
            f"{share:.0%} of your predicted consumption ({predicted.predicted_peak_kwh:.1f} of "
            f"{predicted.predicted_total_kwh:.1f} kWh) falls in the peak period, billed at "
            f"${tariff.peak_rate:.2f}/kWh against ${tariff.offpeak_rate:.2f}/kWh off-peak. That is "
            f"${predicted.predicted_peak_cost:.2f} of your ${predicted.predicted_total_cost:.2f} predicted bill."
        ),
        expected_benefit=_shift_benefit(saving, shiftable_kwh, "half", predicted),
        estimated_saving=saving,
    )


def _rule_moderate_peak_consumption(predicted: PredictedCost, tariff: TariffRates) -> Optional[Recommendation]:
    """Peak share is noticeable but not yet high - lighter-touch nudge."""
    if predicted.predicted_total_kwh < MIN_KWH_FOR_ADVICE:
        return None
    share = _peak_share(predicted)
    if not (MODERATE_PEAK_SHARE_THRESHOLD <= share < HIGH_PEAK_SHARE_THRESHOLD):
        return None

    gap = _rate_gap(tariff)
    shiftable_kwh = predicted.predicted_peak_kwh * MODERATE_PEAK_SHIFT_SHARE
    saving = round(shiftable_kwh * gap, 2) if gap > 0 else None

    return Recommendation(
        rule="moderate_peak_consumption",
        severity=Severity.ADVISORY,
        title="Trim your peak-hour usage",
        message=(
            f"Move one or two high-draw activities to off-peak hours ({_OUTSIDE}), "
            "such as laundry, dishwashing or water heating."
        ),
        explanation=(
            f"{share:.0%} of your predicted {predicted.predicted_total_kwh:.1f} kWh "
            f"({predicted.predicted_peak_kwh:.1f} kWh) falls in the {_WINDOW} peak window. Each of "
            f"those kWh costs ${gap:.2f} more than it would off-peak."
        ),
        expected_benefit=_shift_benefit(saving, shiftable_kwh, "a quarter", predicted),
        estimated_saving=saving,
    )


def _rule_light_peak_consumption(predicted: PredictedCost, tariff: TariffRates) -> Optional[Recommendation]:
    """Peak share is ordinary (between the low and moderate thresholds): a small, optional saving."""
    if predicted.predicted_total_kwh < MIN_KWH_FOR_ADVICE:
        return None
    share = _peak_share(predicted)
    if not (LOW_PEAK_SHARE_THRESHOLD <= share < MODERATE_PEAK_SHARE_THRESHOLD):
        return None
    gap = _rate_gap(tariff)
    if gap <= 0:
        return None  # flat tariff: nothing to gain, and nothing else to say

    shiftable_kwh = predicted.predicted_peak_kwh * LIGHT_PEAK_SHIFT_SHARE
    saving = round(shiftable_kwh * gap, 2)
    return Recommendation(
        rule="light_peak_consumption",
        severity=Severity.INFO,
        title="A small shift would still save",
        message=(
            f"Your peak-hour usage is modest. If one regular task can wait until after "
            f"{short_clock(PEAK_END)} (a dishwasher or laundry cycle, for example), run it then."
        ),
        explanation=(
            f"{share:.0%} of your predicted {predicted.predicted_total_kwh:.1f} kWh "
            f"({predicted.predicted_peak_kwh:.1f} kWh) falls in the {_WINDOW} peak window, at "
            f"${tariff.peak_rate:.2f}/kWh instead of ${tariff.offpeak_rate:.2f}/kWh."
        ),
        expected_benefit=(
            f"About ${saving:.2f} saved over {_window_label(predicted)} if {shiftable_kwh:.1f} kWh "
            f"of peak usage moves to off-peak hours."
        ),
        estimated_saving=saving,
    )


def _shift_benefit(saving: Optional[float], shiftable_kwh: float, fraction: str, predicted: PredictedCost) -> str:
    if saving is None:
        return (
            "No cost saving on your current tariff, because your peak and off-peak rates are the "
            "same. It still eases demand at the busiest time of day."
        )
    return (
        f"About ${saving:.2f} saved over {_window_label(predicted)} if {fraction} of your peak "
        f"usage ({shiftable_kwh:.1f} kWh) moves to off-peak hours."
    )


def _rule_low_peak_consumption(predicted: PredictedCost, tariff: TariffRates) -> Optional[Recommendation]:
    """Positive reinforcement when the household is already mostly off-peak."""
    if predicted.predicted_total_kwh < MIN_KWH_FOR_ADVICE:
        return None
    share = _peak_share(predicted)
    if share >= LOW_PEAK_SHARE_THRESHOLD:
        return None

    return Recommendation(
        rule="already_off_peak",
        severity=Severity.INFO,
        title="Your timing is already good",
        message="Keep running your larger appliances outside peak hours, as you do now.",
        explanation=(
            f"Only {share:.0%} of your predicted {predicted.predicted_total_kwh:.1f} kWh "
            f"falls in the {_WINDOW} peak window."
        ),
        expected_benefit=(
            f"Keeps {1 - share:.0%} of your usage on the lower ${tariff.offpeak_rate:.2f}/kWh rate."
        ),
    )


def _rule_rising_cost(
    predicted: PredictedCost, comparison_cost: Optional[float], comparison_label: str
) -> Optional[Recommendation]:
    """Predicted cost noticeably higher than the same length of actual usage before it."""
    if comparison_cost is None or comparison_cost <= 0:
        return None
    increase = predicted.predicted_total_cost - comparison_cost
    increase_pct = increase / comparison_cost
    if increase_pct < RISING_COST_THRESHOLD:
        return None

    return Recommendation(
        rule="rising_cost_trend",
        severity=Severity.ADVISORY,
        title="Your costs are trending up",
        message=(
            "Check which appliances are running more than usual (heating, cooling and hot water "
            "are the usual causes) and cut back where you can."
        ),
        explanation=(
            f"Your predicted cost for {_window_label(predicted)} is ${predicted.predicted_total_cost:.2f}, "
            f"{increase_pct:.0%} higher than the ${comparison_cost:.2f} you actually spent in {comparison_label}."
        ),
        expected_benefit=f"Up to ${increase:.2f} avoided if usage returns to its recent level.",
    )


def _rule_partial_forecast_caveat(predicted: PredictedCost) -> Optional[Recommendation]:
    """Flag when recommendations are based on incomplete forecast data."""
    if not predicted.is_partial_forecast:
        return None
    return Recommendation(
        rule="partial_forecast_data",
        severity=Severity.INFO,
        title="Based on a partial forecast",
        message="Treat these figures as a lower bound until the full forecast is available.",
        explanation=(
            f"The forecast covers {predicted.covered_hours:.0f} of the "
            f"{predicted.expected_hours:.0f} hours in this window."
        ),
        expected_benefit="Savings estimates will firm up once the whole window is forecast.",
    )


def generate_recommendations(
    predicted: PredictedCost,
    tariff: TariffRates,
    prior_cycle_cost: Optional[float] = None,
    comparison_label: str = "your last billing cycle",
) -> List[Recommendation]:
    """
    Run all rules against the household's predicted cost (US-22 output) and
    tariff, returning every recommendation that fired, highest priority first.

    `prior_cycle_cost` is optional: the actual cost of a comparable earlier
    period (see current_cost.split_cost). It enables the rising-cost rule;
    `comparison_label` says what that period was.
    """
    candidates = [
        _rule_high_peak_consumption(predicted, tariff),
        _rule_moderate_peak_consumption(predicted, tariff),
        _rule_light_peak_consumption(predicted, tariff),
        _rule_rising_cost(predicted, prior_cycle_cost, comparison_label),
        _rule_low_peak_consumption(predicted, tariff),
        _rule_partial_forecast_caveat(predicted),
    ]
    fired = [r for r in candidates if r is not None]
    return sorted(fired, key=lambda r: _PRIORITY_ORDER.index(r.priority))