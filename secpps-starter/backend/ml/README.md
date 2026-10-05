# WattSense ML — anomaly detection

## Run order

Run from the `ml` folder:

```powershell
python run_feature_engineering.py
python run_anomaly_detection.py
```

The forecasting package is independent of this anomaly-detection workflow.

## Detection pipeline

The anomaly pipeline now uses three outputs:

1. **Isolation Forest** — multivariate point/context anomaly detector.
2. **LSTM Autoencoder** — 24-hour temporal reconstruction detector.
3. **Validation-calibrated IF + AE ensemble** — the final alert decision used by the API/dashboard.

The ensemble does not use a fixed manually chosen IF/AE weighting.  During US-13A it evaluates a grid of IF/AE weights on synthetic anomalies injected into the **validation split only**.  For each weight it also searches the decision threshold.  The selected weight and threshold are then locked before the held-out TEST evaluation.

The default ensemble objective is **F2** (`ENSEMBLE_BETA = 2.0`), which gives recall more importance than precision because the current project priority is detecting more true anomalies.  This can be changed in `config.py` without changing the model code.

To reduce sensitivity to one random anomaly placement, validation calibration is repeated three times using deterministic seeds and the validation predictions are pooled before selecting the final ensemble configuration.

## Important output files

```text
outputs/results/validation_threshold_calibration.csv
outputs/results/ensemble_validation_grid.csv
outputs/results/if_results.csv
outputs/results/ae_results.csv
outputs/results/ensemble_results.csv
outputs/results/evaluation_metrics.csv
outputs/results/evaluation_summary.csv
outputs/results/evaluation_by_anomaly_type.csv
outputs/models/if_ae_ensemble.pkl
outputs/graphs/if_ae_ensemble.png
outputs/api/anomaly_api_output.csv
outputs/api/anomaly_api_output.json
```

`evaluation_summary.csv` contains the pooled held-out TEST results for Isolation Forest, LSTM Autoencoder and the IF+AE ensemble.  `evaluation_by_anomaly_type.csv` reports spike/dropout detection recall separately.

## Leakage controls

- Every household is split chronologically into train/validation/test.
- Rolling energy baselines are causal and exclude the current reading.
- Scalers and missing-value medians are fitted on TRAIN only.
- IF/AE base thresholds are calibrated on VALIDATION only.
- Ensemble weights and the ensemble decision threshold are calibrated on VALIDATION only.
- TEST is used only after all detector settings have been locked.

## LSTM settings

The current laptop-friendly defaults are:

```text
Sequence length             : 24 hours
Maximum training sequences  : 75,000
Maximum validation sequences: 20,000
Maximum test sequences      : 30,000
Batch size                  : 32
Maximum epochs              : 20
Early stopping patience     : 3
LSTM units                  : 32 -> 16 -> 16 -> 32
```

The complete multivariate sequence is reconstructed, but the point anomaly error is calculated at the current/final timestep using the energy-sensitive dimensions.  This keeps the anomaly label aligned with the timestamp being evaluated.

## Logs

Pipeline messages are printed to the terminal and written to timestamped files under:

```text
outputs/results/logs/
```

Graph saving is fault-tolerant: a graph-export problem is logged and a fallback path is attempted instead of terminating model evaluation.
