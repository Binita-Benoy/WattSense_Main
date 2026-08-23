# feature_extraction/peak_features.py
import pandas as pd
import numpy as np
from config import PEAK_PERCENTILE, RAW_DATA_PATH

def add_peak_features(df: pd.DataFrame,
                      target_col: str = "kwh",
                      window_days: int = 30) -> pd.DataFrame:
    """
    Add peak-hour identification features.

    Features added:
    - is_peak:    1 if current consumption > rolling 80th percentile
                  0 otherwise
    - peak_ratio: current consumption / rolling daily mean
                  (how many times the average is this hour?)

    Args:
        df:          DataFrame with DatetimeIndex and target_col
        target_col:  name of the consumption column
        window_days: rolling window for percentile calculation

    Returns:
        DataFrame with peak feature columns added
    """
    df = df.copy()
    window = window_days * 24   # convert days to hours

    # Rolling 80th percentile threshold
    rolling_threshold = (
        df[target_col]
        .rolling(window=window, min_periods=window // 4)
        .quantile(PEAK_PERCENTILE / 100)
    )

    # Binary peak flag
    df["is_peak"] = (df[target_col] > rolling_threshold).astype(int)

    # Peak ratio: current / rolling daily mean
    rolling_daily_mean = (
        df[target_col]
        .rolling(window=24, min_periods=6)
        .mean()
    )
    df["peak_ratio"] = (df[target_col] / rolling_daily_mean).round(4)
    df["peak_ratio"] = df["peak_ratio"].clip(upper=10)   # cap extreme ratios

    # Summary statistics
    peak_rate = df["is_peak"].mean() * 100
    print(f"  Peak flag: {peak_rate:.1f}% of hours flagged as peak")
    print(f"  (Expected ~20% given 80th percentile threshold)")

    return df


if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_PATH, index_col="timestamp",
                     parse_dates=True)

    df_peak = add_peak_features(df)

    print("\nSample output:")
    print(df_peak[["kwh", "is_peak", "peak_ratio"]].dropna().head(24))

    print("\nPeak hours by hour of day:")
    peak_by_hour = df_peak.groupby(df_peak.index.hour)["is_peak"].mean() * 100
    print(peak_by_hour.round(1))
    print("(Higher % at evening hours = peak detection working correctly)")