"""Offline smoke tests for the rewritten anomaly pipeline.

Does not require PostgreSQL or TensorFlow. It validates feature causality,
public-holiday propagation, Isolation Forest calibration/evaluation, sequence
context construction, and current-timestep AE error alignment using a dummy
predictor.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from feature_engineering.us10_features import build_feature_matrix
from splitter import split_all_households
from anomaly_detection.us11_isolation_forest import train_isolation_forest
from anomaly_detection.us12_autoencoder import (
    make_sequences,
    reconstruction_errors,
)
from anomaly_detection.us13_evaluation import (
    calibrate_thresholds,
    evaluate_on_injection,
)
from anomaly_detection.ensemble_detector import (
    search_ensemble_parameters,
    predict_ensemble_from_results,
)


def synthetic_raw(n_hours=3200, households=(1001, 1002, 1003, 1004), seed=123):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2013-01-01", periods=n_hours, freq="h")
    frames = []

    for k, hh in enumerate(households):
        hour = index.hour.to_numpy()
        dow = index.dayofweek.to_numpy()
        day = np.arange(n_hours) / 24.0

        temperature = (
            20
            + 7 * np.sin(2 * np.pi * day / 365.0 + 0.3)
            + 2 * np.sin(2 * np.pi * hour / 24 - 0.7)
            + rng.normal(0, 0.7, n_hours)
        )
        humidity = 58 - 0.7 * (temperature - 20) + rng.normal(0, 3, n_hours)
        solar = np.maximum(0, 800 * np.sin(np.pi * (hour - 6) / 12))

        consumption = (
            0.18
            + 0.035 * k
            + 0.24 * np.exp(-0.5 * ((hour - 7) / 1.8) ** 2)
            + 0.42 * np.exp(-0.5 * ((hour - 19) / 2.2) ** 2)
            + 0.05 * (dow >= 5)
            + 0.022 * np.maximum(18 - temperature, 0)
            + 0.028 * np.maximum(temperature - 24, 0)
            + rng.normal(0, 0.045, n_hours)
        )
        consumption = np.clip(consumption, 0.03, None)

        holiday = (
            (index.normalize() == pd.Timestamp("2013-01-26"))
            | (index.normalize() == pd.Timestamp("2013-04-25"))
        ).astype(int)
        consumption = consumption + 0.06 * holiday

        frame = pd.DataFrame(
            {
                "household_id": hh,
                "kwh": consumption,
                "controlled_load_kwh": 0.0,
                "gross_generation_kwh": 0.0,
                "net_generation_kwh": 0.0,
                "other_kwh": 0.0,
                "source": "synthetic",
                "is_public_holiday": holiday,
                "temperature": temperature,
                "humidity": humidity,
                "visibility": 10.0,
                "apparent_temperature": temperature + rng.normal(0, 0.4, n_hours),
                "pressure": 1012 + rng.normal(0, 3, n_hours),
                "wind_speed": 3 + rng.normal(0, 0.6, n_hours),
                "cloud_cover": rng.uniform(0, 1, n_hours),
                "wind_bearing": rng.uniform(0, 360, n_hours),
                "precip_intensity": np.maximum(0, rng.normal(0.05, 0.08, n_hours)),
                "dew_point": temperature - (100 - humidity) / 5,
                "precip_probability": rng.uniform(0, 0.3, n_hours),
                "shortwave_radiation": solar,
            },
            index=index,
        )
        frame.index.name = "timestamp"
        frames.append(frame)

    return pd.concat(frames).sort_index()


def test_causal_rolling_and_holiday():
    raw = synthetic_raw(n_hours=1200, households=(1001,), seed=1)
    F1 = build_feature_matrix(raw, save=False)
    assert "is_public_holiday" in F1.columns
    assert set(F1["is_public_holiday"].unique()).issubset({0, 1})

    # A current reading must not change its own rolling_24h baseline.
    timestamp = F1.index[len(F1) // 2]
    altered = raw.copy()
    altered.loc[timestamp, "kwh"] *= 20
    F2 = build_feature_matrix(altered, save=False)
    a = float(F1.loc[timestamp, "rolling_24h_mean"])
    b = float(F2.loc[timestamp, "rolling_24h_mean"])
    assert np.isclose(a, b, atol=1e-12), (a, b)


def test_if_calibration_and_test_metrics():
    F = build_feature_matrix(synthetic_raw(), save=False)
    splits = split_all_households(F)
    trained = train_isolation_forest(splits["train"])
    calibration = calibrate_thresholds(splits, trained, ae_output=None)
    detail, summary = evaluate_on_injection(splits, trained, ae_output=None)

    assert not calibration.empty
    assert not summary.empty
    row = summary[summary["detector"] == "Isolation Forest"].iloc[0]
    # Smoke-test floor, not a project performance guarantee.
    assert float(row["f1"]) >= 0.75, row.to_dict()
    assert float(row["pr_auc"]) >= 0.75, row.to_dict()


def test_sequence_alignment_without_tensorflow():
    values = np.zeros((30, 3), dtype=np.float32)
    X, endpoints = make_sequences(values, seq_len=24)
    assert X.shape == (7, 24, 3)
    assert endpoints[0] == 23 and endpoints[-1] == 29

    class DummyModel:
        def predict(self, X, batch_size=None, verbose=0):
            pred = X.copy()
            # Large reconstruction error in an EARLIER timestep should not
            # affect the current-point anomaly score.
            pred[0, 3, 0] += 10.0
            # Current/final timestep error should be visible.
            pred[1, -1, 0] += 2.0
            return pred

    errors = reconstruction_errors(DummyModel(), X[:2], [0])
    assert np.isclose(errors[0], 0.0), errors
    assert np.isclose(errors[1], 2.0), errors


def test_ensemble_calibration_and_prediction():
    # Complementary validation scores: IF is strong on one subset while AE is
    # strong on another.  Because the search includes pure IF and pure AE,
    # the selected ensemble F2 cannot be worse than both individual endpoints.
    y = np.array([0] * 80 + [1] * 20, dtype=int)
    rng = np.random.default_rng(99)

    if_scores = np.r_[
        rng.uniform(0.05, 0.45, 80),
        np.r_[rng.uniform(0.70, 0.95, 10), rng.uniform(0.25, 0.55, 10)],
    ]
    ae_scores = np.r_[
        rng.uniform(0.05, 0.45, 80),
        np.r_[rng.uniform(0.25, 0.55, 10), rng.uniform(0.70, 0.95, 10)],
    ]

    best, grid = search_ensemble_parameters(y, if_scores, ae_scores, beta=2.0)
    assert 0.0 <= float(best["weight_if"]) <= 1.0
    assert 0.0 <= float(best["weight_ae"]) <= 1.0
    assert np.isclose(float(best["weight_if"]) + float(best["weight_ae"]), 1.0)
    assert 0.0 <= float(best["threshold"]) <= 1.0
    endpoint_best = grid[grid["weight_ae"].isin([0.0, 1.0])]["fbeta"].max()
    assert float(best["fbeta"]) + 1e-12 >= float(endpoint_best)

    ts = pd.date_range("2014-01-01", periods=6, freq="h")
    if_df = pd.DataFrame(
        {
            "household_id": 1001,
            "consumption": np.linspace(0.2, 0.7, 6),
            "raw_anomaly_score": np.linspace(0.3, 0.8, 6),
            "anomaly_score": [0.1, 0.2, 0.3, 0.6, 0.8, 0.9],
            "is_anomaly": [0, 0, 0, 0, 1, 1],
        },
        index=ts,
    )
    if_df.index.name = "timestamp"
    ae_df = pd.DataFrame(
        {
            "timestamp": ts,
            "household_id": 1001,
            "consumption": np.linspace(0.2, 0.7, 6),
            "recon_error": np.linspace(0.01, 0.20, 6),
            "anomaly_score": [0.05, 0.25, 0.35, 0.75, 0.85, 0.95],
            "is_anomaly": [0, 0, 0, 1, 1, 1],
        }
    )
    cfg = {
        "weight_if": 0.4,
        "weight_ae": 0.6,
        "threshold": 0.55,
        "objective": "F2",
    }
    out = predict_ensemble_from_results(if_df, ae_df, cfg)
    assert len(out) == 6
    assert out["ensemble_score"].between(0, 1).all()
    assert set(out["is_anomaly"].unique()).issubset({0, 1})
    assert int(out["has_both_scores"].sum()) == 6


def main():
    tests = [
        test_causal_rolling_and_holiday,
        test_if_calibration_and_test_metrics,
        test_sequence_alignment_without_tensorflow,
        test_ensemble_calibration_and_prediction,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"PASS all {len(tests)} smoke tests")


if __name__ == "__main__":
    main()
