"""Run the complete WattSense anomaly-detection pipeline."""
from __future__ import annotations

import logging

import pandas as pd

from config import AE_RESULTS_CSV, IF_RESULTS_CSV
from data_utils import load_feature_matrix
from logging_utils import configure_logging, log_print
from splitter import split_all_households
from anomaly_detection.us11_isolation_forest import (
    plot_if_evidence,
    predict_if_on_split,
    train_isolation_forest,
    verify_us11,
)
from anomaly_detection.us12_autoencoder import (
    compute_error_reference,
    plot_ae_evidence,
    predict_dataframe,
    prepare_split_sequences,
    train_autoencoder,
    verify_us12,
)
from anomaly_detection.us13_evaluation import (
    calibrate_thresholds,
    evaluate_on_injection,
    verify_us13,
)
from anomaly_detection.ensemble_detector import (
    load_ensemble_config,
    plot_ensemble_evidence,
    predict_ensemble_from_results,
    save_ensemble_results,
    verify_ensemble,
)
from anomaly_detection.us14_api import (
    build_api_payload,
    save_api_outputs,
    verify_us14,
)

logger = logging.getLogger()


def main():
    logger, log_path = configure_logging("anomaly_detection")
    try:
        log_print("WATTSENSE — ANOMALY DETECTION")
        features = load_feature_matrix()
        splits = split_all_households(features)

        # ------------------------------------------------------------------
        # Isolation Forest is trained on TRAIN only.
        # ------------------------------------------------------------------
        log_print("\n>>> US-11: Train Isolation Forest")
        if_trained = train_isolation_forest(splits["train"])

        # ------------------------------------------------------------------
        # LSTM AE is trained on TRAIN and early-stopped on chronological
        # validation sequences.  Train-percentile threshold is fallback only.
        # ------------------------------------------------------------------
        log_print("\n>>> US-12: Train multivariate LSTM Autoencoder")
        seq_data = prepare_split_sequences(
            splits["train"],
            splits["val"],
            splits["test"],
        )
        ae_model = train_autoencoder(seq_data)
        ae_fallback_threshold = compute_error_reference(ae_model, seq_data)
        ae_output = {
            "model": ae_model,
            "threshold": ae_fallback_threshold,
            "scaler": seq_data["scaler"],
            "feature_cols": seq_data["feature_cols"],
            "sequence_data": seq_data,
        }

        # ------------------------------------------------------------------
        # VALIDATION ONLY.
        # Calibrate IF threshold, AE threshold, ensemble weights and ensemble
        # threshold.  Multiple deterministic validation injection repeats are
        # pooled for more stable calibration.  TEST is not used here.
        # ------------------------------------------------------------------
        log_print("\n>>> US-13A: Validation threshold + ensemble calibration")
        calibrate_thresholds(splits, if_trained, ae_output)

        # ------------------------------------------------------------------
        # Base-detector predictions on clean chronological TEST.
        # ------------------------------------------------------------------
        log_print("\n>>> Final US-11 predictions with calibrated threshold")
        if_results_test = predict_if_on_split(
            splits["test"],
            if_trained,
            split_name="test",
        )
        if_results_test.to_csv(IF_RESULTS_CSV)
        plot_if_evidence(if_results_test, if_trained)
        us11_ok = verify_us11(if_results_test)

        log_print("\n>>> Final US-12 predictions with calibrated threshold")
        test_history = pd.concat([splits["train"], splits["val"]], axis=0)
        ae_results_test = predict_dataframe(
            ae_model,
            splits["test"],
            test_history,
            seq_data,
            split_name="test",
        )
        ae_results_test.to_csv(AE_RESULTS_CSV, index=False)
        plot_ae_evidence(ae_results_test)
        us12_ok = verify_us12(ae_results_test)

        # ------------------------------------------------------------------
        # Final weighted IF + AE ensemble on clean TEST.
        # ------------------------------------------------------------------
        log_print("\n>>> Final validation-calibrated IF+AE ensemble predictions")
        ensemble_config = load_ensemble_config()
        ensemble_results = predict_ensemble_from_results(
            if_results_test,
            ae_results_test,
            ensemble_config,
            split_name="test",
        )
        save_ensemble_results(ensemble_results)
        plot_ensemble_evidence(ensemble_results, ensemble_config)
        ensemble_ok = verify_ensemble(ensemble_results)

        # ------------------------------------------------------------------
        # independently injected held-out TEST evaluation.
        # Reports IF, AE and ensemble using settings already locked on VAL.
        # ------------------------------------------------------------------
        log_print("\n>>> US-13B: Held-out injection evaluation")
        metrics, summary = evaluate_on_injection(
            splits,
            if_trained=if_trained,
            ae_output=ae_output,
        )
        us13_ok = verify_us13(metrics, summary)

        # ------------------------------------------------------------------
        # dashboard/API output uses the ENSEMBLE as the final decision,
        # while retaining individual IF/AE scores and flags for transparency.
        # ------------------------------------------------------------------
        log_print("\n>>> US-14: API Output")
        api_df = build_api_payload()
        save_api_outputs(api_df)
        us14_ok = verify_us14(api_df)

        passed = us11_ok and us12_ok and ensemble_ok and us13_ok and us14_ok
        log_print(f"\nANOMALY DETECTION STATUS: {'PASS' if passed else 'FAIL'}")
        log_print(f"LOG FILE: {log_path}")
        return 0 if passed else 1
    except Exception:
        logger.exception("Pipeline failed")
        log_print(f"ANOMALY DETECTION FAILED. See log: {log_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
