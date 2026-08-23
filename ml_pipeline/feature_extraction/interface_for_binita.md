FEATURE VECTOR INTERFACE
=======================================

HOW TO LOAD MY OUTPUT
─────────────────────
from feature_extraction.pipeline import build_feature_vector
import pandas as pd

# Load clean data
df = pd.read_csv(RAW_DATA_PATH,
                 index_col="timestamp",
                 parse_dates=True)

# Get feature vector
output = build_feature_vector(df)

X      = output["F"]        # feature matrix — your model input
y      = output["y"]        # target variable — what you're predicting
scaler = output["scaler"]   # RobustScaler — needed for inverse transform

SHAPES
──────
X shape: [n_samples, 17]    # 17 features (see column list below)
y shape: [n_samples]        # kWh values

FEATURE COLUMNS (in order)
───────────────────────────
lag_1h, lag_24h, lag_48h, lag_168h   # autoregressive lags
hour_sin, hour_cos                    # time of day (cyclical)
dow_sin, dow_cos                      # day of week (cyclical)
rolling_7d_mean, rolling_7d_std       # 7-day trend
rolling_30d_mean, rolling_30d_std     # 30-day baseline
is_peak, peak_ratio                   # peak hour features
season, month_sin, month_cos          # seasonal features
temperature                           # if weather data available

SCALING
───────
X is already scaled with RobustScaler.
For ARIMA: use output["y"] directly (apply log1p yourself).
For LSTM:  use output["F"] and output["y"] directly.

INVERSE TRANSFORM (to get kWh from scaled predictions)
───────────────────────────────────────────────────────
# When converting scaled predictions back to kWh:
scaler = joblib.load("outputs/lstm_scaler.pkl")
y_pred_kwh = scaler.inverse_transform(y_pred_scaled)

ANOMALY FLAGS
─────────────
Anomaly output (separate from features) — see:
anomaly_detection/output_format.md