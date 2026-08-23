# anomaly_detection/isolation_forest.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

def train_isolation_forest(df_anomaly: pd.DataFrame,
                           contamination: float = 0.05,
                           save_path: str = "outputs/") -> dict:
    """
    Train Isolation Forest for energy consumption anomaly detection.

    contamination = expected proportion of anomalies in the data.
    0.05 means we expect ~5% of hours to be anomalous.

    Args:
        df_anomaly:    prepared anomaly dataset
        contamination: expected fraction of anomalies
        save_path:     where to save the trained model

    Returns:
        dict with model, scaler, predictions, and scores
    """

    # Step 1 — Scale features using RobustScaler
    # (proposal specifies RobustScaler for anomaly detection input)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df_anomaly)

    # Step 2 — Train Isolation Forest
    print("Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=100,          # number of trees
        contamination=contamination,
        random_state=42,
        n_jobs=-1                  # use all CPU cores
    )
    model.fit(X_scaled)

    # Step 3 — Generate predictions
    # sklearn returns: -1 = anomaly, 1 = normal
    # Convert to: 1 = anomaly, 0 = normal (more intuitive)
    raw_predictions = model.predict(X_scaled)
    anomaly_flags = (raw_predictions == -1).astype(int)

    # Step 4 — Generate anomaly scores
    # More negative score = more anomalous
    anomaly_scores = model.score_samples(X_scaled)
    # Normalise to 0–1 range (1 = most anomalous)
    normalised_scores = 1 - (anomaly_scores - anomaly_scores.min()) / \
                           (anomaly_scores.max() - anomaly_scores.min())

    # Step 5 — Build output DataFrame in agreed format
    results = pd.DataFrame({
        "timestamp":  df_anomaly.index,
        "kwh":        df_anomaly.iloc[:, 0].values,
        "flag":       anomaly_flags,        # 1=anomaly, 0=normal
        "score":      normalised_scores.round(4),
        "method":     "isolation_forest"
    }).set_index("timestamp")

    # Step 6 — Summary
    n_anomalies = anomaly_flags.sum()
    print(f"\nIsolation Forest results:")
    print(f"  Total hours evaluated: {len(results):,}")
    print(f"  Anomalies detected:    {n_anomalies:,} "
          f"({n_anomalies/len(results)*100:.1f}%)")
    print(f"  Expected:              ~{contamination*100:.0f}%")

    # Step 7 — Save model and scaler
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(model,  os.path.join(save_path, "isolation_forest.pkl"))
    joblib.dump(scaler, os.path.join(save_path, "anomaly_scaler.pkl"))
    print(f"\nModel saved to {save_path}")

    return {
        "model":       model,
        "scaler":      scaler,
        "results":     results,
        "n_anomalies": n_anomalies
    }


def load_and_predict(new_data: pd.DataFrame,
                     model_path: str = "outputs/isolation_forest.pkl",
                     scaler_path: str = "outputs/anomaly_scaler.pkl") -> pd.DataFrame:
    """
    Load trained model and predict anomalies on new data.
    This is the inference function used by the alert engine.
    """
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X_scaled = scaler.transform(new_data)
    raw_preds = model.predict(X_scaled)
    flags  = (raw_preds == -1).astype(int)
    scores = model.score_samples(X_scaled)

    return pd.DataFrame({
        "flag":   flags,
        "score":  scores.round(4),
        "method": "isolation_forest"
    }, index=new_data.index)


if __name__ == "__main__":
    from prepare_anomaly_data import prepare_anomaly_dataset
    from feature_extraction.cyclical_features import add_cyclical_features
    from config import RAW_DATA_PATH
    import sys
    sys.path.append("..")

    df = pd.read_csv(RAW_DATA_PATH, index_col="timestamp",
                     parse_dates=True)
    df = add_cyclical_features(df)
    df_anomaly = prepare_anomaly_dataset(df)

    output = train_isolation_forest(df_anomaly)
    print("\nSample anomaly output:")
    print(output["results"][output["results"]["flag"] == 1].head(10))