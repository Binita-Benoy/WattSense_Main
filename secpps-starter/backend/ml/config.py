"""Central configuration for WattSense feature engineering and anomaly detection."""
from __future__ import annotations

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

OUT_BASE = os.path.join(BASE_DIR, "outputs")
OUT_FEAT = os.path.join(OUT_BASE, "features")
OUT_MODELS = os.path.join(OUT_BASE, "models")
OUT_RESULTS = os.path.join(OUT_BASE, "results")
OUT_API = os.path.join(OUT_BASE, "api")
OUT_GRAPHS = os.path.join(OUT_BASE, "graphs")
OUT_LOGS = os.path.join(OUT_RESULTS, "logs")

for _directory in [OUT_FEAT, OUT_MODELS, OUT_RESULTS, OUT_API, OUT_GRAPHS, OUT_LOGS]:
    os.makedirs(_directory, exist_ok=True)

# SGSC households used by the project.
HOUSEHOLD_IDS = [
    8162493, 8163939, 10042604, 10042756, 10042934, 10042974,
    10174306, 10257896, 10305996, 10341486, 10346126, 10346582,
    10390745, 10413359, 10508173, 10508205, 10534327, 10534355,
    10673547, 10673789, 10733766, 10733884, 10760980, 10850536,
    10859314, 10883534, 10885220, 11006864, 11228008, 11446117,
]
PRIMARY_HH = 10042974

# Column names.
DB_TIMESTAMP = "reading_timestamp"
DB_KWH = "consumption_kwh"
DB_HOUSEHOLD = "household_id"
COL_TIMESTAMP = "timestamp"
COL_KWH = "kwh"
COL_HOUSEHOLD = "household_id"
COL_PUBLIC_HOLIDAY = "is_public_holiday"

# Chronological train / validation / test split.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Feature engineering.
LAG_HOURS = [1, 24, 48, 168]
ROLLING_WINDOWS = {"7d": 7 * 24, "30d": 30 * 24}
PEAK_PERCENTILE = 80
PEAK_WINDOW_DAYS = 30

# Isolation Forest
IF_CONTAMINATION = "auto"
IF_N_ESTIMATORS = 250
IF_MAX_SAMPLES = 2048
IF_MAX_FEATURES = 0.85
IF_RANDOM_STATE = 42
IF_FALLBACK_THRESHOLD_PCT = 98.5
IF_SCORE_LOW_PCT = 1.0
IF_SCORE_HIGH_PCT = 99.5

# LSTM Autoencoder.
AE_SEQ_LEN = 24
AE_MAX_TRAIN_SEQUENCES = 75_000
AE_MAX_VAL_SEQUENCES = 20_000
AE_MAX_TEST_SEQUENCES = 30_000
AE_BATCH_SIZE = 32
AE_EPOCHS = 20
AE_PATIENCE = 3
AE_LSTM_UNITS_1 = 32
AE_LSTM_UNITS_2 = 16
AE_PREDICT_BATCH_SIZE = 256
AE_RANDOM_STATE = 42
AE_THRESHOLD_PCT = 97.5  # fallback only; validation calibration is preferred
AE_SCORE_LOW_PCT = 5.0
AE_SCORE_HIGH_PCT = 99.5

# Validation-calibrated IF + AE ensemble.
ENSEMBLE_BETA = 2.0
ENSEMBLE_WEIGHT_STEP = 0.05
ENSEMBLE_CALIBRATION_REPEATS = 3

# Online/API inference. 
INFERENCE_HISTORY_DAYS = 35  # legacy request-scoped DB-context mode

API_RECENT_HOURS = 7 * 24
API_MAX_ANOMALIES_PER_HOUSEHOLD = 500
API_REQUIRE_HOURLY_HISTORY = True

# Synthetic anomaly evaluation.
INJECTION_SPIKE_FRACTION = 0.010
INJECTION_DROPOUT_FRACTION = 0.005
INJECTION_SPIKE_MIN_MULT = 4.0
INJECTION_SPIKE_MAX_MULT = 7.0
INJECTION_DROPOUT_MULT = 0.03
INJECTION_MIN_SPIKES = 5
INJECTION_MIN_DROPOUTS = 3
CALIBRATION_RANDOM_STATE = 31415
TEST_INJECTION_RANDOM_STATE = 27182

# Dashboard severity is based on stable train-referenced 0-1 scores.
SEVERITY_HIGH = 0.80
SEVERITY_MEDIUM = 0.50

# Output paths.
FEATURE_CSV = os.path.join(OUT_FEAT, "feature_matrix.csv")
SPLIT_INFO_CSV = os.path.join(OUT_FEAT, "split_info.csv")

IF_MODEL_PKL = os.path.join(OUT_MODELS, "isolation_forest.pkl")
IF_SCALER_PKL = os.path.join(OUT_MODELS, "if_scaler.pkl")
IF_PREPROCESS_PKL = os.path.join(OUT_MODELS, "if_preprocess.pkl")
IF_THRESHOLD_PKL = os.path.join(OUT_MODELS, "if_threshold.pkl")

AE_MODEL_PATH = os.path.join(OUT_MODELS, "ae_model.keras")
AE_SCALER_PKL = os.path.join(OUT_MODELS, "ae_scaler.pkl")
AE_PREPROCESS_PKL = os.path.join(OUT_MODELS, "ae_preprocess.pkl")
AE_THRESHOLD_PKL = os.path.join(OUT_MODELS, "ae_threshold.pkl")

IF_RESULTS_CSV = os.path.join(OUT_RESULTS, "if_results.csv")
AE_RESULTS_CSV = os.path.join(OUT_RESULTS, "ae_results.csv")
EVAL_CSV = os.path.join(OUT_RESULTS, "evaluation_metrics.csv")
EVAL_SUMMARY_CSV = os.path.join(OUT_RESULTS, "evaluation_summary.csv")
CALIBRATION_CSV = os.path.join(OUT_RESULTS, "validation_threshold_calibration.csv")
ENSEMBLE_GRID_CSV = os.path.join(OUT_RESULTS, "ensemble_validation_grid.csv")
ENSEMBLE_RESULTS_CSV = os.path.join(OUT_RESULTS, "ensemble_results.csv")
EVAL_BY_TYPE_CSV = os.path.join(OUT_RESULTS, "evaluation_by_anomaly_type.csv")
ENSEMBLE_CONFIG_PKL = os.path.join(OUT_MODELS, "if_ae_ensemble.pkl")

API_JSON = os.path.join(OUT_API, "anomaly_api_output.json")
API_CSV = os.path.join(OUT_API, "anomaly_api_output.csv")

IF_GRAPH = os.path.join(OUT_GRAPHS, "us11_isolation_forest.png")
AE_GRAPH = os.path.join(OUT_GRAPHS, "us12_autoencoder.png")
EVAL_GRAPH = os.path.join(OUT_GRAPHS, "us13_evaluation.png")
ENSEMBLE_GRAPH = os.path.join(OUT_GRAPHS, "if_ae_ensemble.png")
SPLIT_GRAPH = os.path.join(OUT_GRAPHS, "data_split_overview.png")
