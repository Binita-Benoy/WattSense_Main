# feature_extraction/rolling_features.py
import pandas as pd
from config import ROLLING_WINDOWS, RAW_DATA_PATH

def add_rolling_features(df: pd.DataFrame,
                         target_col: str = "kwh") -> pd.DataFrame:
    """
    Add rolling window mean and standard deviation features.

    Windows:
    - 7-day  (168 hours): captures weekly consumption trend
    - 30-day (720 hours): captures monthly consumption baseline

    Uses min_periods to avoid dropping too many rows at start.
    min_periods = window // 4 means we start getting values
    after 25% of the window is filled rather than waiting
    for the full window.

    Args:
        df:         DataFrame with DatetimeIndex and target_col
        target_col: name of the consumption column

    Returns:
        DataFrame with rolling statistics columns added
    """
    df = df.copy()

    for window in ROLLING_WINDOWS:
        days = window // 24
        prefix = f"rolling_{days}d"

        # Rolling mean — captures the trend level
        df[f"{prefix}_mean"] = (
            df[target_col]
            .rolling(window=window, min_periods=window // 4)
            .mean()
            .round(4)
        )

        # Rolling std — captures the volatility / spread
        df[f"{prefix}_std"] = (
            df[target_col]
            .rolling(window=window, min_periods=window // 4)
            .std()
            .round(4)
        )

        print(f"  Added {prefix}_mean and {prefix}_std "
              f"(window={window}h = {days} days)")

    return df


if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_PATH, index_col="timestamp",
                     parse_dates=True)

    df_rolling = add_rolling_features(df)

    rolling_cols = ["kwh",
                    "rolling_7d_mean",  "rolling_7d_std",
                    "rolling_30d_mean", "rolling_30d_std"]

    print("\nSample output (first non-null rows):")
    print(df_rolling[rolling_cols].dropna().head(5))

    print("\nNull counts:")
    print(df_rolling[rolling_cols[1:]].isna().sum())