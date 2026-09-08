"""
System Configuration: per-household model selection.

Rather than forcing one model architecture on every household, this
evaluates BOTH the plain ARIMA baseline and the weather-enhanced SARIMAX
against each household's own held-out test window, and selects whichever
actually performed better FOR THAT HOUSEHOLD.

CAVEAT: this decision is based on a single 24-hour held-out window per
household. That's a reasonable simplification given project timeline,
but a genuinely robust version would re-evaluate across several
different windows before trusting the selection -- worth stating
explicitly in the report rather than overclaiming confidence.
"""
import json
import os

from sqlalchemy import create_engine

from arima_forecast import generate_forecast
from generate_weather_forecast import generate_weather_forecast

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)


def select_best_model(engine, household_id: int) -> dict:
    arima_result = generate_forecast(engine, household_id)
    sarimax_result = generate_weather_forecast(engine, household_id)

    arima_mae = arima_result["metrics"]["mae"]
    sarimax_mae = sarimax_result["metrics"]["mae"]

    winner = "SARIMAX (weather)" if sarimax_mae < arima_mae else "ARIMA (no weather)"

    return {
        "household_id": household_id,
        "arima_mae": arima_mae,
        "sarimax_mae": sarimax_mae,
        "selected_model": winner,
        "margin_pct": round(abs(arima_mae - sarimax_mae) / arima_mae * 100, 2),
        "note": "Based on a single held-out window -- treat as directional, not definitive, for small margins",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--household-ids", type=str, required=True,
                         help="Comma-separated household IDs, e.g. 10042974,10733884,10346126")
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)
    household_ids = [int(x) for x in args.household_ids.split(",")]

    decisions = []
    for hid in household_ids:
        decision = select_best_model(engine, hid)
        decisions.append(decision)
        print(f"Household {hid}: {decision['selected_model']} "
              f"(ARIMA MAE={decision['arima_mae']}, SARIMAX MAE={decision['sarimax_mae']}, "
              f"margin={decision['margin_pct']}%)")

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "model_selection_decisions.json")
    with open(path, "w") as f:
        json.dump(decisions, f, indent=2)
    print(f"\nSaved model selection table to {path}")
