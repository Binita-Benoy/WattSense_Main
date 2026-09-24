"""
US-23 — Generate energy-saving recommendations
------------------------------------------------
As a household user, I want personalised recommendations based on my
consumption and predicted cost, so that I know what actions I can take to
reduce energy usage or cost.

Acceptance criteria (from the story):
  - Recommendations must be based on actual system information (not
    generic tips) — every recommendation here is generated from the
    household's real predicted cost (US-22) and current tariff, and each
    one carries the concrete numbers that triggered it.
  - Pattern: High predicted peak consumption -> recommend shifting usage
    -> explain why.
  - Rule-based, using predicted cost + current circumstances (not ML).

This module takes the output of billing.py (PredictedCost) and, optionally,
the household's actual current-cycle consumption from current_cost.py, and
runs a small set of rules over them. It does not talk to the DB directly —
callers assemble the inputs and pass them in, which keeps the rules
unit-testable the same way billing.py's calculation logic is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from billing import PredictedCost, TariffRates  # adjust import path once merged into app_services/


class Severity(str, Enum):
    INFO = "info"       # positive reinforcement / no action needed
    ADVISORY = "advisory"
    WARNING = "warning"  # meaningfully high predicted peak cost


@dataclass(frozen=True)
class Recommendation:
    rule: str                 # which rule fired, for traceability/testing
    severity: Severity
    message: str               # the recommendation itself
    explanation: str           # why, grounded in the actual numbers
    estimated_saving: Optional[float] = None  # $ if the shift is followed, when computable


# ---------------------------------------------------------------------------
# Thresholds — kept as named constants (not magic numbers) so they're easy
# to tune later or move into config without touching the rule logic.
# ---------------------------------------------------------------------------
HIGH_PEAK_SHARE_THRESHOLD = 0.55   # peak kWh as a share of total predicted kWh
MODERATE_PEAK_SHARE_THRESHOLD = 0.35
LOW_PEAK_SHARE_THRESHOLD = 0.15
MIN_KWH_FOR_ADVICE = 1.0           # ignore near-zero forecasts (not enough signal)


def _peak_share(predicted: PredictedCost) -> float:
    total = predicted.predicted_total_kwh
    if total <= 0:
        return 0.0
    return predicted.predicted_peak_kwh / total


def _rate_gap(tariff: TariffRates) -> float:
    """How much more expensive peak is than off-peak, per kWh."""
    return max(tariff.peak_rate - tariff.offpeak_rate, 0.0)


# ---------------------------------------------------------------------------
# Rules — each takes the same inputs and returns a Recommendation or None.
# Keeping them as separate small functions makes it easy to add/remove a
# rule, and to unit test each independently.
# ---------------------------------------------------------------------------
def _rule_high_peak_consumption(
    predicted: PredictedCost, tariff: TariffRates
) -> Optional[Recommendation]:
    """High predicted peak consumption -> recommend shifting usage -> explain why."""
    if predicted.predicted_total_kwh < MIN_KWH_FOR_ADVICE:
        return None

    share = _peak_share(predicted)
    if share < HIGH_PEAK_SHARE_THRESHOLD:
        return None

    gap = _rate_gap(tariff)
    # Rough "what if half of peak usage moved to off-peak" saving estimate —
    # a concrete, explainable number rather than a vague suggestion.
    shiftable_kwh = predicted.predicted_peak_kwh * 0.5
    estimated_saving = round(shiftable_kwh * gap, 2) if gap > 0 else None

    return Recommendation(
        rule="high_peak_consumption",
        severity=Severity.WARNING,
        message=(
            "Shift some of your usage out of the 3pm-9pm peak window "
            "(e.g. run appliances like washing machines, dishwashers or "
            "EV charging before 3pm or after 9pm)."
        ),
        explanation=(
            f"{share:.0%} of your predicted consumption "
            f"({predicted.predicted_peak_kwh:.1f} kWh of "
            f"{predicted.predicted_total_kwh:.1f} kWh) falls in the peak "
            f"period, which is billed at ${tariff.peak_rate:.2f}/kWh versus "
            f"${tariff.offpeak_rate:.2f}/kWh off-peak — "
            f"driving ${predicted.predicted_peak_cost:.2f} of your "
            f"${predicted.predicted_total_cost:.2f} predicted bill."
        ),
        estimated_saving=estimated_saving,
    )


def _rule_high_predicted_total_cost(
    predicted: PredictedCost, tariff: TariffRates, prior_cycle_cost: Optional[float]
) -> Optional[Recommendation]:
    """Predicted cost noticeably higher than the prior billing cycle."""
    if prior_cycle_cost is None or prior_cycle_cost <= 0:
        return None

    increase = predicted.predicted_total_cost - prior_cycle_cost
    increase_pct = increase / prior_cycle_cost
    if increase_pct < 0.20:  # only flag a >=20% rise
        return None

    return Recommendation(
        rule="rising_cost_trend",
        severity=Severity.ADVISORY,
        message=(
            "Your predicted bill is trending up compared to your last "
            "billing cycle — worth reviewing which appliances are running "
            "more than usual."
        ),
        explanation=(
            f"Predicted cost for this period is ${predicted.predicted_total_cost:.2f}, "
            f"{increase_pct:.0%} higher than your last cycle's "
            f"${prior_cycle_cost:.2f}."
        ),
    )


def _rule_moderate_peak_consumption(
    predicted: PredictedCost, tariff: TariffRates
) -> Optional[Recommendation]:
    """Peak share is noticeable but not yet high — lighter-touch nudge."""
    if predicted.predicted_total_kwh < MIN_KWH_FOR_ADVICE:
        return None

    share = _peak_share(predicted)
    if not (MODERATE_PEAK_SHARE_THRESHOLD <= share < HIGH_PEAK_SHARE_THRESHOLD):
        return None

    return Recommendation(
        rule="moderate_peak_consumption",
        severity=Severity.ADVISORY,
        message=(
            "Consider moving a few more high-draw activities to off-peak "
            "hours (before 3pm or after 9pm) to trim your bill further."
        ),
        explanation=(
            f"{share:.0%} of your predicted {predicted.predicted_total_kwh:.1f} kWh "
            "usage falls in the peak window — not extreme, but still costing "
            f"more than shifting it would."
        ),
    )


def _rule_low_peak_consumption(
    predicted: PredictedCost,
) -> Optional[Recommendation]:
    """Positive reinforcement when the household is already mostly off-peak."""
    if predicted.predicted_total_kwh < MIN_KWH_FOR_ADVICE:
        return None

    share = _peak_share(predicted)
    if share >= LOW_PEAK_SHARE_THRESHOLD:
        return None

    return Recommendation(
        rule="already_off_peak",
        severity=Severity.INFO,
        message="You're already doing most of your energy usage off-peak — keep it up.",
        explanation=(
            f"Only {share:.0%} of your predicted "
            f"{predicted.predicted_total_kwh:.1f} kWh usage falls in the peak window."
        ),
    )


def _rule_partial_forecast_caveat(
    predicted: PredictedCost,
) -> Optional[Recommendation]:
    """Flag when recommendations are based on incomplete forecast data."""
    if not predicted.is_partial_forecast:
        return None

    return Recommendation(
        rule="partial_forecast_data",
        severity=Severity.INFO,
        message=(
            "These recommendations are based on partial forecast data — "
            "they may change as more of your usage is predicted."
        ),
        explanation=(
            f"Forecast covered {predicted.covered_hours:.1f} of "
            f"{predicted.expected_hours:.1f} expected hours in this window."
        ),
    )


_RULES_NEEDING_PRIOR_COST = {"rising_cost_trend"}


def generate_recommendations(
    predicted: PredictedCost,
    tariff: TariffRates,
    prior_cycle_cost: Optional[float] = None,
) -> List[Recommendation]:
    """
    Run all rules against the household's predicted cost (US-22 output) and
    current tariff, returning every recommendation that fired.

    `prior_cycle_cost` is optional — pass the actual total from
    current_cost.py's last completed billing cycle, if available, to enable
    the rising-cost-trend rule. Rules are independent, so recommendations
    may combine (e.g. a high-peak warning alongside a rising-cost advisory).
    """
    candidates = [
        _rule_high_peak_consumption(predicted, tariff),
        _rule_moderate_peak_consumption(predicted, tariff),
        _rule_low_peak_consumption(predicted),
        _rule_high_predicted_total_cost(predicted, tariff, prior_cycle_cost),
        _rule_partial_forecast_caveat(predicted),
    ]
    return [r for r in candidates if r is not None]