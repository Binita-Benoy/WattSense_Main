"""Leakage-aware energy, calendar and weather feature engineering."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import (
    LAG_HOURS,
    ROLLING_WINDOWS,
    PEAK_PERCENTILE,
    PEAK_WINDOW_DAYS,
    FEATURE_CSV,
    COL_HOUSEHOLD,
    COL_KWH,
    COL_PUBLIC_HOLIDAY,
)
from logging_utils import log_print

logger = logging.getLogger()

WEATHER_BASE = [
    "temperature",
    "humidity",
    "apparent_temperature",
    "pressure",
    "wind_speed",
    "cloud_cover",
    "wind_bearing",
    "precip_intensity",
    "dew_point",
    "precip_probability",
    "shortwave_radiation",
]

WEATHER_FEATURES = WEATHER_BASE + [
    "temperature_24h_mean",
    "temperature_24h_std",
    "temperature_deviation_24h",
    "humidity_24h_mean",
    "humidity_deviation_24h",
    "cooling_degree",
    "heating_degree",
    "solar_generation_context",
]


def add_lag_features(df: pd.DataFrame, target: str = COL_KWH) -> pd.DataFrame:
    """Add historical consumption lags; all are strictly from the past."""
    df = df.copy()
    for lag in LAG_HOURS:
        # for each row, copy the 'target' value from 'lag' number of rows earlier into a new column.
        df[f"lag_{lag}h"] = df[target].shift(lag)
    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-of-day, day-of-week, month, weekend and holiday context."""
    df = df.copy()
    hour = df.index.hour
    dow = df.index.dayofweek
    month = df.index.month

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["hour"] = hour.astype("int8")
    df["day_of_week"] = dow.astype("int8")
    df["is_weekend"] = (dow >= 5).astype("int8")

    # The latest DB dump stores this directly in energy_readings.
    if COL_PUBLIC_HOLIDAY in df.columns:
        df[COL_PUBLIC_HOLIDAY] = (
            df[COL_PUBLIC_HOLIDAY].fillna(0).astype(bool).astype("int8")
        )
    else:
        # Defensive fallback for older extracts; the current project DB should
        # normally contain the column.
        df[COL_PUBLIC_HOLIDAY] = np.int8(0)

    return df


def add_rolling_statistics(df: pd.DataFrame, target: str = COL_KWH) -> pd.DataFrame:
    """
    Add causal rolling baselines.

    The current reading is deliberately excluded via shift(1). This prevents a
    spike/dropout from changing the baseline used to judge that same reading.
    """
    df = df.copy()
    history = df[target].shift(1)

    for name, window in ROLLING_WINDOWS.items():
        min_periods = max(6, window // 4)
        rolling = history.rolling(window, min_periods=min_periods) # when min_periods of data is there, a moving window is created by rolling function of 'window' length which is later used to calculate mean and std
        df[f"rolling_{name}_mean"] = rolling.mean()
        df[f"rolling_{name}_std"] = rolling.std()

    rolling_24h = history.rolling(24, min_periods=6)
    df["rolling_24h_mean"] = rolling_24h.mean()
    df["rolling_24h_std"] = rolling_24h.std()
    return df


def add_peak_indicator(df: pd.DataFrame, target: str = COL_KWH) -> pd.DataFrame:
    """Add causal rolling peak threshold, peak flag and ratio-to-local-baseline."""
    df = df.copy()
    history = df[target].shift(1)
    window = PEAK_WINDOW_DAYS * 24
    min_periods = max(24, window // 4)

    threshold = (
        history.rolling(window, min_periods=min_periods)
        .quantile(PEAK_PERCENTILE / 100.0) # here, from a moving 'window', we calculate 'peak_percentile' value in that window which later acts as threshold to identify peaks
    )
    df["peak_threshold_30d"] = threshold
    df["is_peak"] = (df[target] > threshold).astype("int8")

    baseline_24h = history.rolling(24, min_periods=6).mean()
    df["peak_ratio"] = (
        df[target] / baseline_24h.replace(0, np.nan) # this provides a ratio of how current value is compared to 24-hour average
    ).clip(lower=0, upper=10) # clipping is done to restrict ratio values between 0 and 10
    return df


def add_deviation_features(df: pd.DataFrame, target: str = COL_KWH) -> pd.DataFrame:
    """Add point-wise deviation and change features for anomaly detection."""
    df = df.copy()
    if "rolling_24h_mean" not in df.columns:
        df = add_rolling_statistics(df, target)

    df["deviation_24h"] = df[target] - df["rolling_24h_mean"]
    df["z_score"] = (
        df["deviation_24h"] / df["rolling_24h_std"].replace(0, np.nan)
    ).clip(-10, 10)

    # These are equivalent to comparing with historical lags, but using diff()
    # keeps the feature definition explicit and readable.
    df["change_1h"] = df[target].diff(1)
    df["change_24h"] = df[target].diff(24)
    df["abs_change_1h"] = df["change_1h"].abs()

    # Same-hour ratio is useful for both sudden spikes and sudden dropouts.
    eps = 1e-4 # added in calculations later, to avoid division by zero
    if "lag_24h" in df.columns:
        df["lag24_ratio"] = (
            (df[target] + eps) / (df["lag_24h"] + eps)
        ).clip(lower=0, upper=10)

    return df


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Australian meteorological season label as an integer 0..3."""
    df = df.copy()
    season_map = {
        12: 0, 1: 0, 2: 0,      # summer
        3: 1, 4: 1, 5: 1,       # autumn
        6: 2, 7: 2, 8: 2,       # winter
        9: 3, 10: 3, 11: 3,     # spring
    }
    df["season"] = df.index.month.map(season_map).astype("int8")
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weather context features.

    Only short forward-fills are performed here. Remaining missing values are
    intentionally left as NaN and are imputed later using TRAIN-only medians.
    That avoids using test/validation information during feature engineering.
    """
    df = df.copy()
    available = [c for c in WEATHER_BASE if c in df.columns]
    if not available:
        return df

    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].ffill(limit=6)

    if "temperature" in df.columns:
        temperature = df["temperature"]
        rolling = temperature.rolling(24, min_periods=6)
        df["temperature_24h_mean"] = rolling.mean()
        df["temperature_24h_std"] = rolling.std()
        df["temperature_deviation_24h"] = (
            temperature - df["temperature_24h_mean"]
        )
        df["heating_degree"] = np.maximum(18.0 - temperature, 0.0)
        df["cooling_degree"] = np.maximum(temperature - 24.0, 0.0)

    if "humidity" in df.columns:
        humidity = df["humidity"]
        df["humidity_24h_mean"] = humidity.rolling(24, min_periods=6).mean()
        df["humidity_deviation_24h"] = humidity - df["humidity_24h_mean"]

    if "shortwave_radiation" in df.columns:
        df["solar_generation_context"] = np.log1p(
            df["shortwave_radiation"].clip(lower=0)
        )

    return df


def build_feature_matrix(df_all: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Build features independently for each household, then combine them."""
    log_print("\n" + "=" * 70)
    log_print("US-10: LEAKAGE-AWARE ENERGY + WEATHER FEATURE ENGINEERING")
    log_print("=" * 70)

    all_features: list[pd.DataFrame] = []

    for hh_id in sorted(df_all[COL_HOUSEHOLD].unique()):
        df = (
            df_all[df_all[COL_HOUSEHOLD] == hh_id]
            .copy()
            .sort_index()
        )
        log_print(
            f"HH {hh_id}: {len(df):,} rows | "
            f"{df.index.min()} -> {df.index.max()}"
        )

        df = add_lag_features(df)
        df = add_cyclical_features(df)
        df = add_rolling_statistics(df)
        df = add_peak_indicator(df)
        df = add_deviation_features(df)
        df = add_seasonal_features(df)
        df = add_weather_features(df)

        # 30-day causal baselines require a warm-up period. Weather NaNs are not
        # globally filled here; model preprocessing uses training medians.
        before = len(df)
        df = df.dropna(
            subset=["lag_168h", "rolling_30d_mean", "rolling_30d_std", "z_score"]
        )
        df[COL_HOUSEHOLD] = hh_id
        log_print(f"  usable rows: {len(df):,} ({before - len(df):,} warm-up rows removed)")
        all_features.append(df)

    if not all_features:
        raise ValueError("No household produced a usable feature matrix.")

    F = pd.concat(all_features, axis=0).sort_index()
    numeric_cols = F.select_dtypes(include=[np.number]).columns
    F[numeric_cols] = F[numeric_cols].replace([np.inf, -np.inf], np.nan)

    missing = F[numeric_cols].isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        log_print("Remaining numeric NaNs (handled later with TRAIN-only medians):")
        log_print(missing.head(20).to_string())

    if COL_PUBLIC_HOLIDAY in F.columns:
        log_print(f"Public-holiday rows: {int(F[COL_PUBLIC_HOLIDAY].sum()):,}")

    if save:
        F.to_csv(FEATURE_CSV)
        log_print(f"Saved feature matrix: {FEATURE_CSV}")

    log_print(f"Feature matrix shape: {F.shape}")
    log_print(f"Feature columns excluding household_id: {len(F.columns) - 1}")
    return F


def verify_us10(F: pd.DataFrame) -> bool:
    required = [
        "lag_1h", "lag_24h", "lag_48h", "lag_168h",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "is_weekend", COL_PUBLIC_HOLIDAY,
        "rolling_7d_mean", "rolling_7d_std",
        "rolling_30d_mean", "rolling_30d_std",
        "rolling_24h_mean", "rolling_24h_std",
        "peak_threshold_30d", "is_peak", "peak_ratio",
        "deviation_24h", "z_score", "change_1h", "change_24h", "lag24_ratio",
        "temperature", "humidity", "apparent_temperature", "wind_speed",
        "shortwave_radiation", "temperature_deviation_24h",
        "cooling_degree", "heating_degree",
    ]

    log_print("\nUS-10 ACCEPTANCE CRITERIA")
    passed = True
    for name in required:
        ok = name in F.columns
        log_print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        passed &= ok
    log_print(f"US-10 status: {'PASS' if passed else 'FAIL'}")
    return bool(passed)
