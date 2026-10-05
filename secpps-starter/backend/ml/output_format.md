# Anomaly detection output format

The dashboard/API uses the **validation-calibrated IF + LSTM Autoencoder ensemble** as the final anomaly decision.

## Primary fields

| Field | Type | Meaning |
|---|---|---|
| `household_id` | integer | Household identifier |
| `timestamp` | datetime | Hour being evaluated |
| `consumption` | float | Consumption in kWh |
| `anomaly_score` | float 0..1 | Final weighted ensemble score |
| `is_anomaly` | 0/1 | Final ensemble alert decision |
| `severity` | string | NORMAL / LOW / MEDIUM / HIGH |
| `confidence` | string | Agreement confidence from the base detectors |
| `detector` | string | `if_ae_ensemble` for final alerts |
| `model_evidence` | string | `both`, `if_only`, `ae_only`, or `weighted_combination` |

## Explainability fields

The API also retains:

- `if_score`
- `ae_score`
- `if_flag`
- `ae_flag`
- `weight_if`
- `weight_ae`
- `decision_threshold`

The final score is:

```text
ensemble_score = weight_if * IF_score + weight_ae * AE_score
```

for rows where both scores are available.  The weights and threshold are selected on validation data and then locked.

## Confidence

- `HIGH`: final ensemble alert and both base detectors independently flagged the point.
- `MEDIUM`: final ensemble alert and one base detector independently flagged it.
- `LOW`: weighted ensemble crossed its threshold even though neither standalone threshold was crossed.
- `NORMAL`: no final ensemble alert.
