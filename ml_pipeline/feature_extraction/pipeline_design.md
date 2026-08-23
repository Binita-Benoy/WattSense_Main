FEATURE EXTRACTION PIPELINE
========================================

INPUT
-----
Source:    Preprocessed dataset (sgsc_clean.csv)
Format:    Pandas DataFrame, DatetimeIndex at hourly frequency
Columns:   timestamp (index), kwh (float), temperature (float, optional)
Guarantee: No missing values, no outliers, consistent hourly intervals

TRANSFORMATIONS (in order)
---------------------------
Step 1:  Lag features         — autoregressive memory of the series
Step 2:  Cyclical features    — time-of-day and day-of-week encoding
Step 3:  Rolling statistics   — medium-term trend and variance
Step 4:  Peak-hour flags      — demand spike identification
Step 5:  Seasonal features    — season label and holiday indicator
Step 6:  Weather covariates   — temperature/humidity if available
Step 7:  Drop NaN rows        — lags create NaN at start of series
Step 8:  Schema validation    — confirm all expected columns present

OUTPUT
------
Name:    feature_vector_F(t)
Format:  Pandas DataFrame, same DatetimeIndex as input
Shape:   [n_samples × n_features]
Columns: lag_1h, lag_24h, lag_48h, lag_168h,
         hour_sin, hour_cos, dow_sin, dow_cos,
         rolling_7d_mean, rolling_7d_std,
         rolling_30d_mean, rolling_30d_std,
         is_peak, peak_ratio,
         season, is_holiday,
         temperature (if available)
Target:  kwh (kept as separate Series for Binita)

CONSUMERS
---------
Forecasting model:  Uses F(t) as model input; kwh as prediction target
Anomaly detector: Uses F(t) as anomaly detector input