"""
Tests for US-04. Run with:
    docker compose exec backend pytest ml/forecasting/test_arima_forecast.py -v

Pick a REAL household_id from your database before running — replace
KNOWN_HOUSEHOLD_ID below with one you've confirmed has enough data via:
    SELECT household_id, count(*) FROM energy_readings GROUP BY household_id;
"""
import os
import pytest
from sqlalchemy import create_engine
from arima_forecast import generate_forecast

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://secpps:secpps_dev_password@db:5432/smart_energy"
)

KNOWN_HOUSEHOLD_ID = 10859314


@pytest.fixture
def engine():
    return create_engine(DATABASE_URL)


def test_forecast_has_correct_horizon_length(engine):
    result = generate_forecast(engine, KNOWN_HOUSEHOLD_ID, horizon_hours=24)
    assert len(result["predictions"]) == 24


def test_forecast_predictions_have_timestamps(engine):
    result = generate_forecast(engine, KNOWN_HOUSEHOLD_ID, horizon_hours=24)
    for p in result["predictions"]:
        assert "timestamp" in p
        assert "predicted_kwh" in p
        assert isinstance(p["predicted_kwh"], float)


def test_forecast_timestamps_are_sequential_hourly(engine):
    import pandas as pd
    result = generate_forecast(engine, KNOWN_HOUSEHOLD_ID, horizon_hours=24)
    timestamps = [pd.Timestamp(p["timestamp"]) for p in result["predictions"]]
    for i in range(1, len(timestamps)):
        gap = timestamps[i] - timestamps[i - 1]
        assert gap == pd.Timedelta(hours=1)


def test_metrics_are_present_and_valid(engine):
    result = generate_forecast(engine, KNOWN_HOUSEHOLD_ID, horizon_hours=24)
    metrics = result["metrics"]
    assert "mae" in metrics and "rmse" in metrics and "mape" in metrics
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["mape"] >= 0


def test_forecast_is_reproducible(engine):
    """Same household, same order, same data -> same predictions.
    This is the actual proof for the 'results are reproducible' criterion,
    not just an assumption."""
    result_1 = generate_forecast(engine, KNOWN_HOUSEHOLD_ID, horizon_hours=24)
    result_2 = generate_forecast(engine, KNOWN_HOUSEHOLD_ID, horizon_hours=24)

    preds_1 = [p["predicted_kwh"] for p in result_1["predictions"]]
    preds_2 = [p["predicted_kwh"] for p in result_2["predictions"]]

    assert preds_1 == preds_2, "ARIMA forecast is not reproducible across runs"
    assert result_1["metrics"] == result_2["metrics"]
