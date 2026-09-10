# anomaly_detection/run_all.py
# ─────────────────────────────────────────────────────────────────────────────
# Master runner — US-10 through US-14 with correct split discipline
# ─────────────────────────────────────────────────────────────────────────────
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("\n" + "█"*60)
    print("  WATTSENSE — ANOMALY DETECTION SPRINT")
    print("  Correct train/val/test split throughout")
    print("█"*60)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n▶ Loading all 5 households from database...")
    from anomaly_detection.db_loader import load_all_households
    df_all = load_all_households()

    # ── US-10: Feature engineering ─────────────────────────────────────────
    print("\n" + "▶"*3 + " US-10: Feature Engineering")
    from anomaly_detection.us10_features import (
        build_feature_matrix, verify_us10
    )
    F = build_feature_matrix(df_all, save=True)
    verify_us10(F)

    # ── Chronological split ────────────────────────────────────────────────
    print("\n" + "▶"*3 + " CHRONOLOGICAL SPLIT (70/15/15)")
    from anomaly_detection.splitter import split_all_households
    splits = split_all_households(F)
    # splits["train"] → used to FIT models
    # splits["val"]   → used for early stopping only
    # splits["test"]  → used ONLY for evaluation

    # ── US-11: Isolation Forest ────────────────────────────────────────────
    print("\n" + "▶"*3 + " US-11: Isolation Forest")
    from anomaly_detection.us11_isolation_forest import run_if_pipeline
    if_output = run_if_pipeline(splits)

    # ── US-12: LSTM Autoencoder ────────────────────────────────────────────
    print("\n" + "▶"*3 + " US-12: LSTM Autoencoder")
    from anomaly_detection.us12_autoencoder import run_ae_pipeline
    ae_output = run_ae_pipeline(splits)

    # ── US-13: Evaluation ─────────────────────────────────────────────────
    print("\n" + "▶"*3 + " US-13: Evaluation")
    from anomaly_detection.us13_evaluation import (
        evaluate_on_injection, verify_us13
    )
    df_metrics = evaluate_on_injection(
        splits,
        if_trained   = if_output["trained"],
        ae_model     = ae_output["model"],
        ae_threshold = ae_output["threshold"]
    )
    verify_us13(df_metrics)

    # ── US-14: API output ─────────────────────────────────────────────────
    print("\n" + "▶"*3 + " US-14: API Output")
    from anomaly_detection.us14_api import (
        build_api_payload, save_api_outputs, verify_us14
    )
    df_api = build_api_payload()
    save_api_outputs(df_api)
    verify_us14(df_api)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "█"*60)
    print("  SPRINT COMPLETE")
    print("█"*60)
    print("""
  DATA DISCIPLINE:
    Feature engineering:  all data (no leakage — no targets used)
    Model training:       TRAIN split only
    Early stopping:       VAL split only
    Threshold:            TRAIN reconstruction errors
    Final evaluation:     TEST split only (never seen by model)

  Outputs:
    outputs/features/feature_matrix.csv
    outputs/features/split_info.csv
    outputs/models/isolation_forest.pkl
    outputs/models/ae_model.keras
    outputs/results/if_results.csv     ← test split
    outputs/results/ae_results.csv     ← test split
    outputs/results/evaluation_metrics.csv
    outputs/graphs/data_split_overview.png
    outputs/graphs/us11_isolation_forest.png
    outputs/graphs/us12_autoencoder.png
    outputs/graphs/us13_evaluation.png
    outputs/api/anomaly_api_output.json
    outputs/api/anomaly_api_output.csv
    """)


if __name__ == "__main__":
    main()