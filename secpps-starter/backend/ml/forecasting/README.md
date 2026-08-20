# forecasting — owner: Binita

Responsible for: ARIMA baseline, LSTM, CNN-LSTM, evaluation (MAE/RMSE/MAPE).

## Suggested first file
- `arima_baseline.py` — fit/predict on a single household's series, print
  MAE/RMSE/MAPE on a held-out split

## Input contract
Consumes the feature vectors produced by `ml/features/` (see that README).
Scaler objects (RobustScaler for LSTM/CNN-LSTM, log1p for ARIMA) should be
persisted alongside model weights in `ml/forecasting/artifacts/`.
