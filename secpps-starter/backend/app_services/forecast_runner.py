
import argparse
import json
import math
import os
import sys
import traceback

CNN_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ml", "forecasting", "cnn_bidirectional")
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

    # make the paths absolute, then work from the pipeline's own folder, because the new script
    # finds its saved_models/ and data/ folders relative to itself.
    args.csv, args.out = os.path.abspath(args.csv), os.path.abspath(args.out)
    os.chdir(CNN_DIR)
    # <<< END ADDED

    sys.path.insert(0, CNN_DIR)
    try:
        from live_forecast_bidirectional import (
            run_live_forecast_all_horizons,
            REC_MODEL_PATH,
        )

        if not os.path.exists(REC_MODEL_PATH):
            raise ValueError(
                "The forecasting model files are missing on the server (saved_models/). "
                "Ask the team to generate or share them."
            )

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