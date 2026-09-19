# anomaly_detection/config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration — updated with proper train/val/test splits
# ─────────────────────────────────────────────────────────────────────────────
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

OUT_BASE    = os.path.join(BASE_DIR, "outputs")
OUT_FEAT    = os.path.join(OUT_BASE, "features")
OUT_MODELS  = os.path.join(OUT_BASE, "models")
OUT_RESULTS = os.path.join(OUT_BASE, "results")
OUT_API     = os.path.join(OUT_BASE, "api")
OUT_GRAPHS  = os.path.join(OUT_BASE, "graphs")
OUT_LOGS    = os.path.join(OUT_RESULTS, "logs")

for d in [OUT_FEAT, OUT_MODELS, OUT_RESULTS, OUT_API, OUT_GRAPHS, OUT_LOGS]:
    os.makedirs(d, exist_ok=True)

# ── Household IDs from Alka's SGSC database ───────────────────────────────────
HOUSEHOLD_IDS = [8162493,
8163939,
10042604,
10042756,
10042934,
10042974,
10174306,
10257896,
10305996,
10341486,
10346126,
10346582,
10390745,
10413359,
10508173,
10508205,
10534327,
10534355,
10673547,
10673789,
10733766,
10733884,
10760980,
10850536,
10859314,
10883534,
10885220,
11006864,
11228008,
11446117]
PRIMARY_HH    = 10042974

# ── Column names ─────────────────────────────────────────────────────────────
DB_TIMESTAMP  = "reading_timestamp"
DB_KWH        = "consumption_kwh"
DB_HOUSEHOLD  = "household_id"
COL_TIMESTAMP = "timestamp"
COL_KWH       = "kwh"
COL_HOUSEHOLD = "household_id"

# ── Chronological Train / Val / Test split ────────────────────────────────────
# Splitting by time not by random index — critical for time series
# Train: model learns what NORMAL looks like
# Val:   used only for LSTM early stopping
# Test:  held out completely — evaluated last, once
#
# Proportions: 70% / 15% / 15%
# Concrete dates derived from SGSC range (2011-10 to 2014-03)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
# TEST_RATIO is implicitly 1 - TRAIN_RATIO - VAL_RATIO

# ── Feature engineering settings ─────────────────────────────────────────────
LAG_HOURS        = [1, 24, 48, 168]
ROLLING_WINDOWS  = {"7d": 7*24, "30d": 30*24}
PEAK_PERCENTILE  = 80
PEAK_WINDOW_DAYS = 30

# ── Isolation Forest ──────────────────────────────────────────────────────────
IF_CONTAMINATION = 0.05
IF_N_ESTIMATORS  = 100
IF_RANDOM_STATE  = 42

# ── LSTM Autoencoder ──────────────────────────────────────────────────────────
AE_SEQ_LEN              = 24
# Laptop-friendly settings. These cap the number of sequences used by the AE
# without changing the underlying 24-hour multivariate approach.
AE_MAX_TRAIN_SEQUENCES = 100_000
AE_MAX_VAL_SEQUENCES   = 20_000
AE_MAX_TEST_SEQUENCES  = 30_000
AE_BATCH_SIZE          = 16
AE_EPOCHS              = 20
AE_THRESHOLD_PCT       = 95
AE_PATIENCE             = 3
AE_LSTM_UNITS_1        = 32
AE_LSTM_UNITS_2        = 16
AE_PREDICT_BATCH_SIZE  = 256
AE_RANDOM_STATE        = 42

# ── Severity thresholds ───────────────────────────────────────────────────────
SEVERITY_HIGH   = 0.80
SEVERITY_MEDIUM = 0.50

# ── Output paths ──────────────────────────────────────────────────────────────
FEATURE_CSV      = os.path.join(OUT_FEAT,    "feature_matrix.csv")
SPLIT_INFO_CSV   = os.path.join(OUT_FEAT,    "split_info.csv")
IF_MODEL_PKL     = os.path.join(OUT_MODELS,  "isolation_forest.pkl")
IF_SCALER_PKL    = os.path.join(OUT_MODELS,  "if_scaler.pkl")
AE_MODEL_PATH    = os.path.join(OUT_MODELS,  "ae_model.keras")
AE_SCALER_PKL    = os.path.join(OUT_MODELS,  "ae_scaler.pkl")
AE_THRESHOLD_PKL = os.path.join(OUT_MODELS,  "ae_threshold.pkl")
IF_RESULTS_CSV   = os.path.join(OUT_RESULTS, "if_results.csv")
AE_RESULTS_CSV   = os.path.join(OUT_RESULTS, "ae_results.csv")
EVAL_CSV         = os.path.join(OUT_RESULTS, "evaluation_metrics.csv")
API_JSON         = os.path.join(OUT_API,     "anomaly_api_output.json")
API_CSV          = os.path.join(OUT_API,     "anomaly_api_output.csv")
IF_GRAPH         = os.path.join(OUT_GRAPHS,  "us11_isolation_forest.png")
AE_GRAPH         = os.path.join(OUT_GRAPHS,  "us12_autoencoder.png")
EVAL_GRAPH       = os.path.join(OUT_GRAPHS,  "us13_evaluation.png")
SPLIT_GRAPH      = os.path.join(OUT_GRAPHS,  "data_split_overview.png")