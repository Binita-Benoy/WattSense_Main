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

for d in [OUT_FEAT, OUT_MODELS, OUT_RESULTS, OUT_API, OUT_GRAPHS]:
    os.makedirs(d, exist_ok=True)

# ── Household IDs from Alka's SGSC database ───────────────────────────────────
HOUSEHOLD_IDS = [10042974, 10733884, 10346126, 10859314, 11228008]
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
AE_SEQ_LEN       = 24
AE_BATCH_SIZE    = 32
AE_EPOCHS        = 50
AE_THRESHOLD_PCT = 95

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