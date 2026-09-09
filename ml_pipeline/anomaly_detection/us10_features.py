# anomaly_detection/us10_features.py
# ─────────────────────────────────────────────────────────────────────────────
# US-10: Create energy consumption features
#
# Acceptance criteria — all features from proposal:
# ✅ 1-hour lag
# ✅ 24-hour lag
# ✅ 48-hour lag
# ✅ 168-hour lag
# ✅ hour/day cyclical features
# ✅ rolling statistics
# ✅ peak-hour indicator
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_detection.config import (
    LAG_HOURS, ROLLING_WINDOWS, PEAK_PERCENTILE,
    PEAK_WINDOW_DAYS, FEATURE_CSV, COL_HOUSEHOLD, COL_KWH
)


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL FEATURE FUNCTIONS
# Each can be tested independently
# ─────────────────────────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame,
                     target: str = COL_KWH) -> pd.DataFrame:
    """
    US-10 AC: 1-hour, 24-hour, 48-hour, 168-hour lags.

    Lags encode temporal autocorrelation:
      lag_1h:   immediate momentum — high now → likely high soon
      lag_24h:  same hour yesterday → captures daily rhythm
      lag_48h:  same hour 2 days ago → reinforces daily pattern
      lag_168h: same hour last week → captures weekly seasonality

    Processed per household so lags do not bleed
    across different customers' data.
    """
    df = df.copy()
    for lag in LAG_HOURS:
        df[f"lag_{lag}h"] = df[target].shift(lag)
    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    US-10 AC: hour/day cyclical features.

    Why sine-cosine not raw integers?
    SGSC data has strong 18:00-21:00 evening peaks.
    Hour 23 and hour 0 are 1 hour apart in reality,
    but raw integers give distance of 23.
    Sine-cosine encoding gives distance ≈ 0.

    Encodes:
      hour of day  (period 24)
      day of week  (period 7)
      month        (period 12)
    """
    df    = df.copy()
    hour  = df.index.hour
    dow   = df.index.dayofweek
    month = df.index.month

    df["hour_sin"]   = np.sin(2 * np.pi * hour  / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * hour  / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * dow   / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * dow   / 7)
    df["month_sin"]  = np.sin(2 * np.pi * month / 12)
    df["month_cos"]  = np.cos(2 * np.pi * month / 12)
    df["hour"]       = hour
    df["day_of_week"]= dow
    df["is_weekend"] = (dow >= 5).astype(int)
    return df


def add_rolling_statistics(df: pd.DataFrame,
                           target: str = COL_KWH) -> pd.DataFrame:
    """
    US-10 AC: rolling statistics.

    7-day and 30-day rolling mean and std.
    Also 24-hour rolling for deviation calculations.

    min_periods = window // 4 avoids excessive NaN
    at the start of each household's series.
    """
    df = df.copy()

    # 7-day and 30-day (from proposal)
    for name, window in ROLLING_WINDOWS.items():
        min_p = window // 4
        df[f"rolling_{name}_mean"] = (
            df[target].rolling(window, min_periods=min_p).mean()
        )
        df[f"rolling_{name}_std"]  = (
            df[target].rolling(window, min_periods=min_p).std()
        )

    # 24-hour (needed for deviation and z-score)
    df["rolling_24h_mean"] = (
        df[target].rolling(24, min_periods=6).mean()
    )
    df["rolling_24h_std"]  = (
        df[target].rolling(24, min_periods=6).std()
    )
    return df


def add_peak_indicator(df: pd.DataFrame,
                       target: str = COL_KWH) -> pd.DataFrame:
    """
    US-10 AC: peak-hour indicator.

    Proposal spec: rolling 80th-percentile threshold.

    Why rolling not fixed?
    SGSC data spans 2011-2014, across Australian seasons.
    Winter baseline is higher than summer — a fixed threshold
    would flag winter evenings as peak even when they are normal
    for that season. Rolling 80th percentile adapts automatically.
    """
    df     = df.copy()
    window = PEAK_WINDOW_DAYS * 24
    min_p  = window // 4

    threshold = (
        df[target]
        .rolling(window, min_periods=min_p)
        .quantile(PEAK_PERCENTILE / 100)
    )
    df["is_peak"]    = (df[target] > threshold).astype(int)
    rolling_daily    = df[target].rolling(24, min_periods=6).mean()
    df["peak_ratio"] = (
        (df[target] / rolling_daily).clip(upper=10).round(4)
    )
    return df


def add_deviation_features(df: pd.DataFrame,
                           target: str = COL_KWH) -> pd.DataFrame:
    """
    Deviation from 24h baseline and z-score.
    These are the primary signals for anomaly detection.
    """
    df = df.copy()

    if "rolling_24h_mean" not in df.columns:
        df = add_rolling_statistics(df, target)

    df["deviation_24h"] = df[target] - df["rolling_24h_mean"]
    df["z_score"]       = (
        df["deviation_24h"]
        / df["rolling_24h_std"].replace(0, np.nan)
    ).clip(-10, 10)
    df["change_1h"]     = df[target].diff(1)
    df["change_24h"]    = df[target].diff(24)
    return df


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Season label — Southern Hemisphere (Australia).
    SGSC is from Sydney (NSW), so seasons are reversed
    relative to Northern Hemisphere datasets.
    """
    season_map = {
        12:0, 1:0, 2:0,    # Summer (Dec-Feb)
         3:1, 4:1, 5:1,    # Autumn (Mar-May)
         6:2, 7:2, 8:2,    # Winter (Jun-Aug)
         9:3, 10:3, 11:3   # Spring (Sep-Nov)
    }
    df           = df.copy()
    df["season"] = df.index.month.map(season_map)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MASTER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_matrix(df_all: pd.DataFrame,
                         save: bool = True) -> pd.DataFrame:
    """
    Build complete feature matrix for all 5 households.

    IMPORTANT: each household is processed independently.
    Rolling windows and lags must NOT bleed across households
    — a lag from household A should never see data from B.

    Pipeline:
      1. Split by household
      2. Apply all feature functions
      3. Drop NaN (from lags and rolling windows)
      4. Recombine
      5. Save

    Args:
        df_all: combined DataFrame with household_id column
        save:   whether to write feature_matrix.csv

    Returns:
        Feature DataFrame ready for US-11 and US-12
    """
    print("\n" + "="*60)
    print("US-10: FEATURE ENGINEERING")
    print("="*60)

    households   = df_all[COL_HOUSEHOLD].unique()
    all_features = []

    for hh_id in sorted(households):
        print(f"\n  Processing household {hh_id}...")

        # Extract single household — sort by time
        df = df_all[df_all[COL_HOUSEHOLD] == hh_id].copy()
        df = df.sort_index()

        # Only keep kwh for feature engineering
        # (controlled_load_kwh kept as extra context)
        has_ctrl = "controlled_load_kwh" in df.columns

        print(f"    Rows: {len(df):,} | "
              f"{df.index.min().date()} → "
              f"{df.index.max().date()}")

        # Apply feature functions in order
        df = add_lag_features(df)
        df = add_cyclical_features(df)
        df = add_rolling_statistics(df)
        df = add_peak_indicator(df)
        df = add_deviation_features(df)
        df = add_seasonal_features(df)

        # Drop rows where critical features are NaN
        # lag_168h requires 168 rows; rolling_30d requires 720 rows
        n_before = len(df)
        df = df.dropna(subset=["lag_168h", "rolling_30d_mean"])
        n_after  = len(df)

        print(f"    After dropna: {n_after:,} rows "
              f"({n_before - n_after} dropped)")

        # Tag with household ID
        df[COL_HOUSEHOLD] = hh_id
        all_features.append(df)

    # Combine all households
    F = pd.concat(all_features, axis=0)
    F = F.sort_index()

    print(f"\n{'='*60}")
    print(f"FEATURE MATRIX COMPLETE")
    print(f"{'='*60}")
    print(f"  Shape:      {F.shape}")
    print(f"  Households: {sorted(F[COL_HOUSEHOLD].unique())}")
    feature_names = [c for c in F.columns
                     if c != COL_HOUSEHOLD]
    print(f"  Features ({len(feature_names)}):")
    for f in feature_names:
        print(f"    {f}")

    # Verify no unexpected nulls
    null_counts = F.drop(columns=[COL_HOUSEHOLD]).isna().sum()
    null_counts = null_counts[null_counts > 0]
    if null_counts.empty:
        print(f"\n  Null check: PASSED ✅")
    else:
        print(f"\n  Remaining nulls:")
        print(null_counts)

    if save:
        F.to_csv(FEATURE_CSV)
        print(f"\n  Saved: {FEATURE_CSV}")

    return F


# ─────────────────────────────────────────────────────────────────────────────
# ACCEPTANCE CRITERIA VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def verify_us10(F: pd.DataFrame) -> bool:
    """Print US-10 acceptance criteria check."""
    print(f"\n{'='*60}")
    print("US-10 ACCEPTANCE CRITERIA")
    print("="*60)

    criteria = [
        ("1-hour lag",          "lag_1h"          in F.columns),
        ("24-hour lag",         "lag_24h"         in F.columns),
        ("48-hour lag",         "lag_48h"         in F.columns),
        ("168-hour lag",        "lag_168h"        in F.columns),
        ("Hour sin (cyclical)", "hour_sin"        in F.columns),
        ("Hour cos (cyclical)", "hour_cos"        in F.columns),
        ("Day sin (cyclical)",  "dow_sin"         in F.columns),
        ("Day cos (cyclical)",  "dow_cos"         in F.columns),
        ("Rolling 7d mean",     "rolling_7d_mean" in F.columns),
        ("Rolling 7d std",      "rolling_7d_std"  in F.columns),
        ("Rolling 30d mean",    "rolling_30d_mean"in F.columns),
        ("Rolling 30d std",     "rolling_30d_std" in F.columns),
        ("Peak-hour indicator", "is_peak"         in F.columns),
    ]

    all_pass = True
    for name, passed in criteria:
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {name}")
        if not passed:
            all_pass = False

    status = "ALL PASSED ✅" if all_pass else "SOME FAILED ❌"
    print(f"\n  {status}")
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from anomaly_detection.db_loader import load_all_households
    df_all = load_all_households()
    F      = build_feature_matrix(df_all)
    verify_us10(F)
    print(f"\nSample (3 rows, transposed):")
    print(F.head(3).T.to_string())