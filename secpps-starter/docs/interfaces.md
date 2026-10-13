# Shared interfaces (edit this together, week 1 — don't let it go stale)

Fill this in during your 30-min sync. Whoever changes a contract here should
ping the consuming teammate(s) before merging.

## 1. Cleaned reading (Alka → Mishael/Binita)
```json
{ "household_id": "H0001", "ts": "2026-08-19T14:00:00+10:00", "kwh": 1.243 }
```

## 2. Feature vector F(t) (Mishael → Binita)
Columns: TBD — see backend/ml/features/README.md, finalise together.

## 3. Forecast output (Binita → Abrein/Jazim)
```json
{
  "household_id": "H0001",
  "horizon": "1h",
  "predicted_kwh": 1.31,
  "confidence_low": 1.05,
  "confidence_high": 1.58,
  "model": "lstm_v1"
}
```

## 4. Anomaly flag (Mishael → Abrein/Jazim)
See backend/ml/anomaly/README.md.

## 5. Auth (Jazim → everyone)
JWT in `Authorization: Bearer <token>` header. Payload shape: TBD.
Payload shape:
{ "sub": "<user_email>", "exp": <unix_timestamp> }

To protect any endpoint, import and use the dependency:
    from api.auth import get_current_user
    ...
    def my_route(current_user: User = Depends(get_current_user)):
        ...
