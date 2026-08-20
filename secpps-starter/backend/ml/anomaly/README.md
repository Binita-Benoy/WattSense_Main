# anomaly — owner: Mishael

Responsible for: LSTM autoencoder (reconstruction-error based) + Isolation
Forest as a secondary point-anomaly detector.

## Output contract (must not change without telling Abrein/Jazim — feeds alerts)
```json
{
  "household_id": "H0001",
  "ts": "2026-08-19T14:00:00+10:00",
  "is_anomaly": true,
  "score": 0.87,
  "detector": "autoencoder"
}
```
