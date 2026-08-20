# features — owner: Mishael

Responsible for: turning cleaned `energy_readings` rows into feature vector F(t)
— lag (1h/24h/48h/168h), cyclical hour/day-of-week, 7d/30d rolling stats,
peak-hour flag, season/holiday indicators, weather covariates.

## Suggested first file
- `build_features.py` — takes a pandas DataFrame of cleaned readings, returns
  a feature DataFrame

## Output contract (must not change without telling Binita)
One row per (household_id, ts) with columns:

`lag_1h, lag_24h, lag_48h, lag_168h, hour_sin, hour_cos, dow_sin, dow_cos,
roll_mean_7d, roll_std_7d, roll_mean_30d, roll_std_30d, is_peak_hour,
season, is_holiday, temp_c, humidity_pct, solar_irradiance`
