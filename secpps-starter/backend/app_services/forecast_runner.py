"""
Runs the new-user forecasting pipeline in its OWN process.

Why a separate process: the pipeline loads TensorFlow, fine-tunes two models and
takes a while. Running it here keeps that memory and any crash away from the web
server, and lets the API kill it if it runs too long.

Usage (the API calls this, you normally don't):
    python forecast_runner.py --csv input.csv --location "Sydney, Australia" \
        --country AU --state NSW --out result.json

Writes {"ok": true, "result": {...}} and exits 0, or {"ok": false, "error": "..."}
and exits 2 for a failure we can describe. Progress lines are printed to stdout.
"""
import argparse
import json
import math
import os
import sys
import traceback

# live_pipeline.py imports its sibling modules by plain name (features,
# simple_cnn_lstm, direct_multistep_global), so its folder must be on sys.path.
CNN_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ml", "forecasting", "cnn")
)


def _clean(obj):
    """Replace NaN/inf with None so the result is valid JSON for the browser."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _friendly(exc: Exception) -> str:
    try:
        import requests

        if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            return ("We couldn't reach the weather service (Open-Meteo). "
                    "Please check the internet connection and try again.")
        if isinstance(exc, requests.exceptions.HTTPError):
            return f"The weather service returned an error ({exc}). Please try again shortly."
    except ImportError:
        pass
    if isinstance(exc, ValueError):
        return str(exc)  # the pipeline's own checks, e.g. "Could not geocode location"
    return f"Forecast failed ({type(exc).__name__}: {exc})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--location", required=True)
    parser.add_argument("--country", default="AU")
    parser.add_argument("--state", default="NSW")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sys.path.insert(0, CNN_DIR)
    try:
        from live_pipeline import run_live_forecast_all_horizons

        result = run_live_forecast_all_horizons(
            user_csv_path=args.csv,
            location=args.location,
            country_code=args.country,
            state=args.state,
        )
        payload, code = {"ok": True, "result": _clean(result)}, 0
    except Exception as exc:  # noqa: BLE001 - we report every failure to the user
        traceback.print_exc()
        payload, code = {"ok": False, "error": _friendly(exc)}, 2

    with open(args.out, "w") as f:
        json.dump(payload, f, allow_nan=False)
    sys.exit(code)


if __name__ == "__main__":
    main()