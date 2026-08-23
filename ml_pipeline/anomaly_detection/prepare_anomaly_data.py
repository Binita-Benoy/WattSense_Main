# anomaly_detection/prepare_anomaly_data.py
import pandas as pd
import numpy as np

def prepare_anomaly_dataset(df: pd.DataFrame,
                            target_col: str = "kwh") -> pd.DataFrame:
    """
    Prepare features specifically for anomaly detection.

    Anomaly detection uses different features than forecasting:
    - Forecasting needs sequential lag structure
    - Anomaly detection needs features that capture what
      'normal' looks like so deviations stand out

    Features for anomaly detection:
    - kwh:              raw consumption value
    - hour_sin/cos:     time of day (cyclical)
    - dow_sin/cos:      day of week (cyclical)
    - rolling_24h_mean: recent daily baseline
    - deviation:        how far current value is from recent mean
    - z_score:          standardised deviation from rolling mean/std

    Args:
        df:         DataFrame with DatetimeIndex, target_col,
                    and cyclical features already added
        target_col: consumption column name

    Returns:
        DataFrame ready for Isolation Forest training
    """
    df = df.copy()

    # Recent baseline (24-hour rolling mean)
    df["rolling_24h_mean"] = (
        df[target_col].rolling(24, min_periods=6).mean()
    )
    df["rolling_24h_std"] = (
        df[target_col].rolling(24, min_periods=6).std()
    )

    # Deviation from recent mean
    df["deviation"] = df[target_col] - df["rolling_24h_mean"]

    # Z-score: how many standard deviations from recent mean
    df["z_score"] = (
        (df[target_col] - df["rolling_24h_mean"])
        / df["rolling_24h_std"].replace(0, np.nan)
    ).round(4)

    # Hour of day (raw — useful alongside cyclical for anomaly)
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek

    # Select anomaly features
    anomaly_features = [
        target_col,
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "rolling_24h_mean",
        "deviation",
        "z_score",
        "hour",
        "day_of_week"
    ]

    # Keep only columns that exist
    available = [c for c in anomaly_features if c in df.columns]
    df_anomaly = df[available].dropna()

    print(f"Anomaly dataset prepared:")
    print(f"  Rows: {len(df_anomaly):,}")
    print(f"  Features: {list(df_anomaly.columns)}")
    print(f"  Z-score range: {df_anomaly['z_score'].min():.2f} "
          f"to {df_anomaly['z_score'].max():.2f}")

    return df_anomaly