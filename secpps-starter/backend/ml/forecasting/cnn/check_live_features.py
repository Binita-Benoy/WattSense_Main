"""Find where the NaNs in the live pipeline come from (no training, runs in seconds)."""
import sys
import pandas as pd
from live_pipeline import geocode_location, fetch_historical_weather, build_live_feature_vector

csv_path = sys.argv[1] if len(sys.argv) > 1 else "/app/ml/forecasting/synthetic_household.csv"
user_df = pd.read_csv(csv_path)
user_df["timestamp"] = pd.to_datetime(user_df["timestamp"])
print(f"CSV: {len(user_df)} rows, {user_df['timestamp'].min()} -> {user_df['timestamp'].max()}, "
      f"kwh NaNs={user_df['kwh'].isna().sum()}")

geo = geocode_location("Sydney")
w = fetch_historical_weather(geo["lat"], geo["lon"],
                             user_df["timestamp"].min().strftime("%Y-%m-%d"),
                             user_df["timestamp"].max().strftime("%Y-%m-%d"))
print(f"\nWeather: {len(w)} rows, {w['timestamp'].min()} -> {w['timestamp'].max()}")
print("Weather NaNs per column:\n", w.isna().sum().to_string())
matched = user_df["timestamp"].isin(w["timestamp"]).sum()
print(f"CSV timestamps matching a weather timestamp: {matched}/{len(user_df)}")

f = build_live_feature_vector(user_df, w)
nan_cols = f.isna().sum()
print(f"\nFeature table: {len(f)} rows")
print("Columns with NaNs:\n", nan_cols[nan_cols > 0].to_string() if nan_cols.any() else "  none")
