# WattSense Anomaly Detection Pipeline

## Running the pipeline

Run these commands from the `ml` folder:

```powershell
python run_feature_engineering.py
python run_anomaly_detection.py
```

## Terminal output + log file

Pipeline messages are sent to **both**:

1. the terminal, using `print()` through `log_print()`;
2. a timestamped log file under:

```text
outputs/results/logs/
```

For example:

```text
outputs/results/logs/anomaly_detection_20260917_001500.log
```

The same message is therefore visible while the program is running and preserved for later review.

## Laptop-friendly LSTM Autoencoder

US-12 remains a multivariate 24-hour LSTM autoencoder and keeps the selected energy, temporal and weather features. The workload is reduced using configurable sequence caps and a smaller LSTM:

```text
Maximum training sequences : 100,000
Maximum validation sequences: 20,000
Maximum test sequences      : 30,000
Batch size                  : 16
Maximum epochs              : 20
Early stopping patience     : 3
LSTM units                  : 32 -> 16 -> 16 -> 32
```

The sequence caps are deterministic and retain representation from all configured households. They can be changed in `config.py` without changing the rest of the pipeline.

### US-12 training output

Each epoch is explicitly printed and logged, for example:

```text
Epoch 01/20 | loss=0.012345 | val_loss=0.010876
Epoch 02/20 | loss=0.009876 | val_loss=0.009321
```

This avoids the large memory cost of materialising the previous ~432k training sequences while still training the same type of multivariate LSTM autoencoder.

## Important TensorFlow note

On native Windows, recent TensorFlow versions do not provide the normal CUDA GPU path. The autoencoder can still train using the CPU. The reduced US-12 configuration is intended to make this practical on a laptop.
