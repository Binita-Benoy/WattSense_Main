"""
Runs the COMPLETE comparison: ARIMA vs SARIMAX vs full-feature LSTM,
at BOTH the 24-hour and 7-day horizons, across every real household --
one command instead of running ~30 separate ones by hand.

Uses the TUNED LSTM config if you pass --lstm-lookback/--lstm-units
(from lstm_hyperparameter_tuning.py's output for that household);
otherwise falls back to sensible defaults.
"""
import json
import os

from sqlalchemy import create_engine, text

from arima_forecast import generate_forecast as arima_24h
from generate_weather_forecast import generate_weather_forecast as sarimax_24h
from generate_weekly_weather_forecast import (
    generate_weekly_plain_forecast as sarima_7d,
    generate_weekly_weather_forecast as sarimax_7d,
)
from lstm_full_features import generate_full_feature_forecast as lstm_24h
from lstm_full_features_weekly import generate_full_feature_weekly_forecast as lstm_7d

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)


def discover_real_households(engine, min_hours: int = 2000, limit: int = None) -> list:
    query = """
        SELECT h.household_id, count(DISTINCT e.reading_timestamp) AS n_hours
        FROM households h
        JOIN energy_readings e ON e.household_id = h.household_id
        WHERE h.source NOT IN ('synthetic', 'homec_kaggle', 'demo_house')
        GROUP BY h.household_id
        HAVING count(DISTINCT e.reading_timestamp) >= :min_hours
        ORDER BY h.household_id
    """
    with engine.connect() as conn:
        result = conn.execute(text(query), {"min_hours": min_hours})
        ids = [row[0] for row in result]
    return ids[:limit] if limit else ids


def run_full_comparison(engine, household_ids: list, lstm_lookback: int = 24, lstm_units: int = 32) -> list:
    all_results = []

    for hid in household_ids:
        print(f"\n{'='*60}\nHousehold {hid}\n{'='*60}")
        row = {"household_id": hid}

        print("[24h] ARIMA...")
        try:
            row["arima_24h"] = arima_24h(engine, hid)["metrics"]
            print(f"  {row['arima_24h']}")
        except Exception as e:
            row["arima_24h"] = {"error": str(e)}

        print("[24h] SARIMAX...")
        try:
            row["sarimax_24h"] = sarimax_24h(engine, hid)["metrics"]
            print(f"  {row['sarimax_24h']}")
        except Exception as e:
            row["sarimax_24h"] = {"error": str(e)}

        print("[24h] LSTM (full features)...")
        try:
            row["lstm_24h"] = lstm_24h(engine, hid, lookback=lstm_lookback, units=lstm_units)["metrics"]
            print(f"  {row['lstm_24h']}")
        except Exception as e:
            row["lstm_24h"] = {"error": str(e)}

        print("[7d] SARIMA (no weather)...")
        try:
            row["sarima_7d"] = sarima_7d(engine, hid)["metrics"]
            print(f"  {row['sarima_7d']}")
        except Exception as e:
            row["sarima_7d"] = {"error": str(e)}

        print("[7d] SARIMAX (weather)...")
        try:
            row["sarimax_7d"] = sarimax_7d(engine, hid)["metrics"]
            print(f"  {row['sarimax_7d']}")
        except Exception as e:
            row["sarimax_7d"] = {"error": str(e)}

        print("[7d] LSTM (full features)...")
        try:
            row["lstm_7d"] = lstm_7d(engine, hid)["metrics"]
            print(f"  {row['lstm_7d']}")
        except Exception as e:
            row["lstm_7d"] = {"error": str(e)}

        all_results.append(row)

    return all_results


def print_summary_table(all_results: list):
    print(f"\n\n{'='*100}")
    print("SUMMARY -- MAE by model and horizon")
    print(f"{'='*100}")
    header = f"{'Household':<12}{'ARIMA 24h':<12}{'SARIMAX 24h':<14}{'LSTM 24h':<12}{'SARIMA 7d':<12}{'SARIMAX 7d':<14}{'LSTM 7d':<12}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        def mae(key):
            v = r.get(key, {})
            return f"{v.get('mae', 'ERR'):.4f}" if isinstance(v.get("mae"), (int, float)) else "ERR"
        print(f"{r['household_id']:<12}{mae('arima_24h'):<12}{mae('sarimax_24h'):<14}{mae('lstm_24h'):<12}"
              f"{mae('sarima_7d'):<12}{mae('sarimax_7d'):<14}{mae('lstm_7d'):<12}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--household-ids", type=str, default=None,
                         help="Comma-separated; omit to auto-discover all real households")
    parser.add_argument("--lstm-lookback", type=int, default=24)
    parser.add_argument("--lstm-units", type=int, default=32)
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)

    if args.household_ids:
        household_ids = [int(x) for x in args.household_ids.split(",")]
    else:
        household_ids = discover_real_households(engine)
        print(f"Auto-discovered households: {household_ids}")

    all_results = run_full_comparison(engine, household_ids, args.lstm_lookback, args.lstm_units)
    print_summary_table(all_results)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "full_model_comparison.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {path}")
