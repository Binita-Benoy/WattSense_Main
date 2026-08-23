# config.py
import os

# Paths
DATA_DIR        = "data/"
OUTPUT_DIR      = "outputs/"
RAW_DATA_PATH   = os.path.join(DATA_DIR, "sgsc_clean.csv")

# Feature settings
LAG_HOURS       = [1, 24, 48, 168]
ROLLING_WINDOWS = [7 * 24, 30 * 24]   # 7 days and 30 days in hours
PEAK_PERCENTILE = 80                   # rolling 80th percentile threshold
SEASON_MAP      = {12: 0, 1: 0, 2: 0, # summer (Southern Hemisphere)
                   3:  1, 4: 1, 5: 1, # autumn
                   6:  2, 7: 2, 8: 2, # winter
                   9:  3, 10:3, 11:3} # spring

# Anomaly settings
ISOLATION_FOREST_CONTAMINATION = 0.05   # expected 5% anomaly rate
ANOMALY_OUTPUT_FORMAT = {
    "timestamp":  "ISO 8601",
    "flag":       "int (0=normal, 1=anomaly)",
    "score":      "float (anomaly score)",
    "method":     "str (isolation_forest)"
}