# feature_extraction/pipeline.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from config import LAG_HOURS, ROLLING_WINDOWS, PEAK_PERCENTILE, SEASON_MAP, RAW_DATA_PATH

from lag_features      import add_lag_features
from cyclical_features import add_cyclical_features
from rolling_features  import add_rolling_features
from peak_features     import add_peak_features

def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add season label and cyclical month encoding."""
    df = df.copy()
    month = df.index.month

    # Season label (Southern Hemisphere)
    df["season"] = month.map(SEASON_MAP)

    # Cyclical month encoding
    df["month_sin"] = np.sin(2 * np.pi * month / 12).round(6)
    df["month_cos"] = np.cos(2 * np.pi * month / 12).round(6)

    return df


def build_feature_vector(df: pd.DataFrame,
                         target_col: str = "kwh",
                         scale: bool = True,
                         save_scaler: bool = True,
                         scaler_path: str = "outputs/") -> dict:
    """
    Master pipeline: raw clean DataFrame → scaled feature vector F(t).

    This is the single function called in forecasting pipeline to get model-ready data.

    Args:
        df:           Clean DataFrame (DatetimeIndex, kwh column)
        target_col:   Consumption column name
        scale:        Whether to apply RobustScaler to feature matrix
        save_scaler:  Whether to persist the fitted scaler
        scaler_path:  Where to save scaler object

    Returns:
        dict with:
          - F:       scaled feature DataFrame (model input)
          - y:       target Series (kwh values)
          - scaler:  fitted RobustScaler object
          - columns: list of feature names
    """
    print("=" * 50)
    print("FEATURE EXTRACTION PIPELINE")
    print("=" * 50)

    # Step 1 — Lag features
    print("\nStep 1: Adding lag features...")
    df = add_lag_features(df, target_col)

    # Step 2 — Cyclical time features
    print("\nStep 2: Adding cyclical time features...")
    df = add_cyclical_features(df)

    # Step 3 — Rolling statistics
    print("\nStep 3: Adding rolling statistics...")
    df = add_rolling_features(df, target_col)

    # Step 4 — Peak hour features
    print("\nStep 4: Adding peak-hour features...")
    df = add_peak_features(df, target_col)

    # Step 5 — Seasonal features
    print("\nStep 5: Adding seasonal features...")
    df = add_seasonal_features(df)

    # Step 6 — Weather covariates (if available)
    weather_cols = []
    if "temperature" in df.columns:
        weather_cols = ["temperature"]
        print("\nStep 6: Weather covariates found — including temperature")
    else:
        print("\nStep 6: No weather data — skipping weather covariates")

    # Step 7 — Drop rows with NaN (introduced by lags and rolling windows)
    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    print(f"\nStep 7: Dropped {n_before - n_after} NaN rows "
          f"({n_before} → {n_after})")

    # Step 8 — Separate features (X) and target (y)
    feature_cols = (
        [f"lag_{h}h" for h in LAG_HOURS] +
        ["hour_sin", "hour_cos", "dow_sin", "dow_cos"] +
        ["rolling_7d_mean", "rolling_7d_std",
         "rolling_30d_mean", "rolling_30d_std"] +
        ["is_peak", "peak_ratio"] +
        ["season", "month_sin", "month_cos"] +
        weather_cols
    )

    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    print(f"\nStep 8: Feature matrix shape: {X.shape}")
    print(f"  Features: {feature_cols}")

    # Step 9 — Scale features
    scaler = None
    if scale:
        print("\nStep 9: Applying RobustScaler...")
        scaler = RobustScaler()
        X_scaled_values = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled_values,
                         columns=feature_cols,
                         index=X.index)

        if save_scaler:
            os.makedirs(scaler_path, exist_ok=True)
            joblib.dump(scaler,
                        os.path.join(scaler_path, "lstm_scaler.pkl"))
            print(f"  Scaler saved to {scaler_path}lstm_scaler.pkl")

    # Step 10 — Schema validation
    print("\nStep 10: Validating output schema...")
    assert len(X) == len(y), "X and y must have same length"
    assert not X.isna().any().any(), "Feature matrix contains NaN"
    assert not y.isna().any(), "Target contains NaN"
    print(f"  ✓ Schema valid — {len(X):,} samples, "
          f"{len(feature_cols)} features")

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)

    return {
        "F":       X,
        "y":       y,
        "scaler":  scaler,
        "columns": feature_cols
    }


if __name__ == "__main__":
    # Full pipeline test
    df = pd.read_csv(RAW_DATA_PATH,
                     index_col="timestamp",
                     parse_dates=True)

    output = build_feature_vector(df)

    print("\nFeature vector sample (first 3 rows):")
    print(output["F"].head(3).T)   # transposed for readability

    print(f"\nTarget variable sample:")
    print(output["y"].head(3))

    # Save output
    output["F"].to_csv("outputs/feature_vector_F.csv")
    output["y"].to_csv("outputs/target_y.csv")
    print("\nOutputs saved to outputs/")