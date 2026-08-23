# feature_extraction/lag_features.py
import pandas as pd
from config import LAG_HOURS, RAW_DATA_PATH

def add_lag_features(df: pd.DataFrame, target_col: str = "kwh") -> pd.DataFrame:
    """
    Add autoregressive lag features to the DataFrame.

    Lags:
    - lag_1h:   1 hour ago  — captures immediate momentum
    - lag_24h:  24 hours ago — captures same hour yesterday (daily pattern)
    - lag_48h:  48 hours ago — captures same hour 2 days ago
    - lag_168h: 168 hours ago — captures same hour last week (weekly pattern)

    Args:
        df:         DataFrame with DatetimeIndex and target_col
        target_col: name of the consumption column

    Returns:
        DataFrame with lag columns added
    """
    df = df.copy()

    for lag in LAG_HOURS:
        col_name = f"lag_{lag}h"
        df[col_name] = df[target_col].shift(lag)
        print(f"  Added {col_name} — shifts series back {lag} hour(s)")

    return df


if __name__ == "__main__":
    # Quick test
    import pandas as pd
    df = pd.read_csv(RAW_DATA_PATH, index_col="timestamp",
                     parse_dates=True)

    df_with_lags = add_lag_features(df)

    print("\nSample output (first 5 non-null rows):")
    lag_cols = [f"lag_{h}h" for h in LAG_HOURS]
    print(df_with_lags[["kwh"] + lag_cols].dropna().head())

    print(f"\nNaN rows introduced by lags: {df_with_lags[lag_cols].isna().any(axis=1).sum()}")
    print("(These will be dropped in the final pipeline)")