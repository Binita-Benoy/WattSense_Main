# WattSense US-14 – DB + automatic weather inference

## Intended live-demo flow

1. Insert a new household and hourly consumption history into `households` and `energy_readings`.
2. Call US-14 with the household ID (or IDs).
3. US-14 calls `forecasting.ensure_weather.ensure_historical_weather()` for each household.
4. The teammate weather helper fills `weather_readings` from Open-Meteo when coverage is missing.
5. US-14 reloads the complete energy + weather history from PostgreSQL.
6. US-10 rebuilds the training-compatible causal features.
7. Saved US-11 IF and US-12 LSTM-AE artifacts score the household; US-13 saved ensemble calibration produces the final decision.
8. US-14 returns JSON-ready data for the frontend.

No model is retrained in US-14.

## API request examples

Single household:

```json
{"household_id": 99999999}
```

Several households:

```json
{"household_ids": [99999999, 99999998]}
```

All households currently in `energy_readings`:

```json
{"all_households": true}
```

Only all synthetic households:

```json
{"all_households": true, "source": "synthetic"}
```

Optional controls:

```json
{
  "household_id": 99999999,
  "include_normal": false,
  "ensure_weather": true,
  "weather_required": false,
  "strict_hourly": true,
  "recent_hours": 168,
  "max_anomalies_per_household": 500
}
```

## FastAPI example

```python
from fastapi import FastAPI, HTTPException
from anomaly_detection.us14_api import get_inference_service

app = FastAPI()
anomaly_service = get_inference_service()  # load model artifacts once

@app.post("/api/anomalies/detect")
def detect_anomalies(payload: dict):
    try:
        return anomaly_service.detect_request(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
```

## Frontend interpretation

- `is_anomaly`: authoritative final decision (`1` anomaly, `0` normal).
- `anomaly_score`: anomaly score in the 0–1 display scale; **not a probability**.
- `severity`: `NORMAL`, `LOW`, `MEDIUM`, or `HIGH`.
- `latest_by_household`: easiest source for current household status cards.
- `recent_readings`: bounded history for charts.
- `anomalies`: recent anomaly list/alerts.
- `if_score`, `ae_score`, `if_flag`, `ae_flag`: supporting model evidence.
- `weight_if`, `weight_ae`, `decision_threshold`: saved validation calibration used for final inference.
- `weather_status`: whether the automatic weather step succeeded for each household.
- `weather_coverage`: actual temperature/humidity/shortwave coverage after loading from PostgreSQL.

The frontend should use `is_anomaly`; it should not invent its own score threshold.

## Required project changes

- Replace `anomaly_detection/us14_api.py` with v6.
- Replace `db_loader.py` with v6 (adds arbitrary-household DB helpers).
- Keep the teammate's `forecasting/ensure_weather.py` unchanged.
- No forecasting source code is modified by this update.

## Weather failure behaviour

By default `weather_required=false`. If Open-Meteo or the network fails, US-14 records a warning and continues. Missing model features are handled by the training-only preprocessing medians saved with the models.

Set `weather_required=true` if the endpoint must fail whenever historical weather cannot be ensured.

## Local execution order for the synthetic demo

```bash
python forecasting/generate_synthetic_household.py --household-id 99999999
```

Then call the API endpoint. A separate weather command is no longer required for anomaly inference because US-14 calls `ensure_historical_weather()` automatically.
