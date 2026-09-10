ANOMALY DETECTION OUTPUT FORMAT
==========================================

DESCRIPTION
The anomaly detection module produces a structured output
for every hour of consumption data evaluated. This output
is consumed by:
  - Alert engine (triggers push notifications)
  - Forecasting pipeline (context for predictions)
  - Dashboard (visualisation of anomaly events)

OUTPUT SCHEMA
─────────────
Field:     timestamp
Type:      datetime (ISO 8601) — "2013-08-15T18:00:00"
Purpose:   Identifies which hour the anomaly occurred

Field:     kwh
Type:      float
Purpose:   Raw consumption value at that timestamp

Field:     flag
Type:      int (0 or 1)
Values:    0 = normal consumption
           1 = anomalous consumption
Purpose:   Primary signal for alert engine

Field:     score
Type:      float (0.0 – 1.0)
Values:    Higher score = more anomalous
           score > 0.7 = high confidence anomaly
           score 0.4–0.7 = moderate confidence
           score < 0.4 = marginal / low confidence
Purpose:   Allows alert engine to filter by severity

Field:     method
Type:      str
Values:    "isolation_forest" | "autoencoder"
Purpose:   Identifies which detector produced the flag
           (two detectors run in parallel)

EXAMPLE OUTPUT
──────────────
timestamp            kwh    flag  score  method
2013-08-15 18:00:00  4.82   1     0.91   isolation_forest
2013-08-15 19:00:00  4.61   1     0.85   isolation_forest
2013-08-16 02:00:00  2.94   1     0.73   isolation_forest
2013-09-01 07:00:00  0.12   1     0.88   isolation_forest

API ENDPOINT
──────────────────────────────────
GET /api/anomalies/latest
Returns: last 24 hours of anomaly flags

GET /api/anomalies/range?start=YYYY-MM-DD&end=YYYY-MM-DD
Returns: anomaly flags for specified date range

ALERT TRIGGER RULE
──────────────────
Alert engine should trigger notification when:
  flag == 1 AND score >= 0.7

Lower threshold (score >= 0.4) can be used for
dashboard warning indicators without push notification.