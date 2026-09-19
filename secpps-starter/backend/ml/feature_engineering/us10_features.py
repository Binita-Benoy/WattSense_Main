from logging_utils import log_print
import logging
logger = logging.getLogger()
# US-10: Energy + weather feature engineering for anomaly detection.
import pandas as pd
import numpy as np

from config import (
    LAG_HOURS, ROLLING_WINDOWS, PEAK_PERCENTILE, PEAK_WINDOW_DAYS,
    FEATURE_CSV, COL_HOUSEHOLD, COL_KWH
)

WEATHER_BASE = [
    "temperature", "humidity", "apparent_temperature", "pressure",
    "wind_speed", "cloud_cover", "wind_bearing", "precip_intensity",
    "dew_point", "precip_probability", "shortwave_radiation"
]

WEATHER_FEATURES = WEATHER_BASE + [
    "temperature_24h_mean", "temperature_24h_std", "temperature_deviation_24h",
    "humidity_24h_mean", "humidity_deviation_24h",
    "cooling_degree", "heating_degree", "solar_generation_context"
]


def add_lag_features(df, target=COL_KWH):
    df = df.copy()
    for lag in LAG_HOURS:
        df[f"lag_{lag}h"] = df[target].shift(lag)
    return df


def add_cyclical_features(df):
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
    df["hour"] = hour
    df["day_of_week"] = dow
    df["is_weekend"] = (dow >= 5).astype(int)

    if "is_public_holiday" in df.columns:
        df["is_public_holiday"] = (
            df["is_public_holiday"]
            .fillna(0)
            .astype(int)
        )
    else:
        df["is_public_holiday"] = 0

    return df


def add_rolling_statistics(df, target=COL_KWH):
    df = df.copy()
    for name, window in ROLLING_WINDOWS.items():
        min_p = window // 4
        df[f"rolling_{name}_mean"] = df[target].rolling(window, min_periods=min_p).mean()
        df[f"rolling_{name}_std"] = df[target].rolling(window, min_periods=min_p).std()
    df["rolling_24h_mean"] = df[target].rolling(24, min_periods=6).mean()
    df["rolling_24h_std"] = df[target].rolling(24, min_periods=6).std()
    return df


def add_peak_indicator(df, target=COL_KWH):
    df = df.copy()
    window = PEAK_WINDOW_DAYS * 24
    min_p = window // 4
    threshold = df[target].rolling(window, min_periods=min_p).quantile(PEAK_PERCENTILE / 100)
    df["is_peak"] = (df[target] > threshold).astype(int)
    rolling_daily = df[target].rolling(24, min_periods=6).mean()
    df["peak_ratio"] = (df[target] / rolling_daily.replace(0, np.nan)).clip(upper=10)
    return df


def add_deviation_features(df, target=COL_KWH):
    df = df.copy()
    if "rolling_24h_mean" not in df.columns:
        df = add_rolling_statistics(df, target)
    df["deviation_24h"] = df[target] - df["rolling_24h_mean"]
    df["z_score"] = (
        df["deviation_24h"] / df["rolling_24h_std"].replace(0, np.nan)
    ).clip(-10, 10)
    df["change_1h"] = df[target].diff(1)
    df["change_24h"] = df[target].diff(24)
    # A simple ramp indicator: large absolute hourly changes are informative.
    df["abs_change_1h"] = df["change_1h"].abs()
    return df


def add_seasonal_features(df):
    df = df.copy()
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                  6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    df["season"] = df.index.month.map(season_map)
    return df


def add_weather_features(df):
    """Create weather context and weather-demand interaction features."""
    df = df.copy()
    available = [c for c in WEATHER_BASE if c in df.columns]
    if not available:
        return df

    # Ensure numeric and interpolate only within each household (caller gives one HH).
    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].interpolate(method="time", limit=6)
        df[col] = df[col].ffill(limit=6).bfill(limit=6)

    if "temperature" in df:
        t = df["temperature"]
        df["temperature_24h_mean"] = t.rolling(24, min_periods=6).mean()
        df["temperature_24h_std"] = t.rolling(24, min_periods=6).std()
        df["temperature_deviation_24h"] = t - df["temperature_24h_mean"]
        # Demand-response context. 18C and 24C are neutral reference points.
        df["heating_degree"] = np.maximum(18.0 - t, 0.0)
        df["cooling_degree"] = np.maximum(t - 24.0, 0.0)

    if "humidity" in df:
        h = df["humidity"]
        df["humidity_24h_mean"] = h.rolling(24, min_periods=6).mean()
        df["humidity_deviation_24h"] = h - df["humidity_24h_mean"]

    if "shortwave_radiation" in df:
        # Context feature: daylight/solar availability. Not a generation estimate.
        df["solar_generation_context"] = np.log1p(df["shortwave_radiation"].clip(lower=0))

    return df


def build_feature_matrix(df_all, save=True):
    log_print("\n" + "=" * 70)
    log_print("US-10: ENERGY + WEATHER FEATURE ENGINEERING")
    log_print("=" * 70)

    all_features = []
    for hh_id in sorted(df_all[COL_HOUSEHOLD].unique()):
        log_print(f"\nProcessing household {hh_id}...")
        df = df_all[df_all[COL_HOUSEHOLD] == hh_id].copy().sort_index()
        log_print(f"Rows: {len(df):,} | {df.index.min()} → {df.index.max()}")

        df = add_lag_features(df)
        df = add_cyclical_features(df)
        df = add_rolling_statistics(df)
        df = add_peak_indicator(df)
        df = add_deviation_features(df)
        df = add_seasonal_features(df)
        df = add_weather_features(df)

        # Require core energy/history features. Weather is retained even when some
        # source weather fields are missing; weather imputation handles short gaps.
        n_before = len(df)
        df = df.dropna(subset=["lag_168h", "rolling_30d_mean", "z_score"])
        log_print(f"After core dropna: {len(df):,} rows ({n_before - len(df)} dropped)")
        df[COL_HOUSEHOLD] = hh_id
        all_features.append(df)

    F = pd.concat(all_features, axis=0).sort_index()

    if "is_public_holiday" in F.columns:
        holiday_rows = int(F["is_public_holiday"].sum())
        log_print(f"Public holiday rows in feature matrix: {holiday_rows:,}")

    # Final numeric cleanup. This avoids model failures caused by empty weather columns.
    numeric_cols = F.select_dtypes(include=[np.number]).columns
    F[numeric_cols] = F[numeric_cols].replace([np.inf, -np.inf], np.nan)
    remaining = F[numeric_cols].isna().sum()
    remaining = remaining[remaining > 0]
    if not remaining.empty:
        log_print("Remaining numeric missing values after weather processing:")
        log_print(remaining.to_string())
        F[numeric_cols] = F[numeric_cols].fillna(F[numeric_cols].median())

    if save:
        F.to_csv(FEATURE_CSV)
        log_print(f"Saved feature matrix: {FEATURE_CSV}")

    log_print(f"\nFeature matrix shape: {F.shape}")
    log_print(f"Total features excluding household_id: {len(F.columns) - 1}")
    return F


def verify_us10(F):
    required = [
        "lag_1h", "lag_24h", "lag_48h", "lag_168h",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "is_weekend", "is_public_holiday",
        "rolling_7d_mean", "rolling_7d_std",
        "rolling_30d_mean", "rolling_30d_std",
        "is_peak", "deviation_24h", "z_score",
        "temperature", "humidity", "apparent_temperature", "wind_speed",
        "shortwave_radiation", "temperature_deviation_24h",
        "cooling_degree", "heating_degree"
    ]
    log_print("\nUS-10 ACCEPTANCE CRITERIA")
    passed = True
    for name in required:
        ok = name in F.columns
        log_print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        passed &= ok
    log_print(f"US-10 status: {'PASS' if passed else 'FAIL'}")
    return bool(passed)
