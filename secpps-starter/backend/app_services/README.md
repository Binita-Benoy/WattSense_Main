# app_services — owner: Abrein

Responsible for: recommendation engine (rule-based + ML scoring), billing
estimator (tariff model + forecasted usage), alert trigger logic.

## Suggested first files
- `billing.py` — takes forecasted kWh + a tariff config, returns projected
  monthly cost
- `recommendations.py` — rule engine consuming forecasts + anomaly flags
