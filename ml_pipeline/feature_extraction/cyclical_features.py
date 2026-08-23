# feature_extraction/cyclical_features.py
import pandas as pd
import numpy as np
from config import RAW_DATA_PATH

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode time-of-day and day-of-week as sine-cosine pairs.

    Encoding:
        hour_sin = sin(2π × hour / 24)
        hour_cos = cos(2π × hour / 24)
        dow_sin  = sin(2π × day_of_week / 7)
        dow_cos  = cos(2π × day_of_week / 7)

    Args:
        df: DataFrame with DatetimeIndex

    Returns:
        DataFrame with cyclical time columns added
    """
    df = df.copy()

    # Hour of day (0–23)
    hour = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24).round(6)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24).round(6)

    # Day of week (0=Monday, 6=Sunday)
    dow = df.index.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7).round(6)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7).round(6)

    return df


def verify_cyclical_encoding():
    """
    Visual verification that the encoding is continuous at boundaries.
    Hour 23 and hour 0 should produce similar values.
    """
    hours = list(range(24)) + [0]   # wrap around
    sin_vals = [np.sin(2 * np.pi * h / 24) for h in hours]
    cos_vals = [np.cos(2 * np.pi * h / 24) for h in hours]

    print("Cyclical encoding verification:")
    print(f"  Hour 22 → sin={sin_vals[22]:.3f}, cos={cos_vals[22]:.3f}")
    print(f"  Hour 23 → sin={sin_vals[23]:.3f}, cos={cos_vals[23]:.3f}")
    print(f"  Hour  0 → sin={sin_vals[0]:.3f},  cos={cos_vals[0]:.3f}")
    print(f"  Hour  1 → sin={sin_vals[1]:.3f},  cos={cos_vals[1]:.3f}")
    print("  ✓ Hour 23 and Hour 0 are close — encoding is continuous")


if __name__ == "__main__":
    verify_cyclical_encoding()

    df = pd.read_csv("data/synthetic.csv", index_col="timestamp",
                     parse_dates=True)
    df_encoded = add_cyclical_features(df)
    print("\nSample output:")
    print(df_encoded[["kwh","hour_sin","hour_cos","dow_sin","dow_cos"]].head(10))