import logging
logger = logging.getLogger()
"""Run US-11 to US-14 with file-only logging."""
import traceback
from logging_utils import configure_logging, capture_terminal_output
from data_utils import load_feature_matrix
from splitter import split_all_households
from anomaly_detection.us11_isolation_forest import run_if_pipeline
from anomaly_detection.us12_autoencoder import run_ae_pipeline
from anomaly_detection.us13_evaluation import evaluate_on_injection, verify_us13
from anomaly_detection.us14_api import build_api_payload, save_api_outputs, verify_us14


def main():
    logger, log_path = configure_logging("anomaly_detection")
    with capture_terminal_output(log_path):
        try:
            logger.info("WATTSENSE — ANOMALY DETECTION")
            features = load_feature_matrix()
            splits = split_all_households(features)
            logger.info(">>> US-11: Isolation Forest")
            if_output = run_if_pipeline(splits)
            logger.info(">>> US-12: Multivariate LSTM Autoencoder")
            ae_output = run_ae_pipeline(splits)
            logger.info(">>> US-13: Evaluation")
            metrics = evaluate_on_injection(
                splits,
                if_trained=if_output["trained"],
                ae_model=ae_output["model"],
                ae_threshold=ae_output["threshold"],
            )
            us13_ok = verify_us13(metrics)
            logger.info(">>> US-14: API Output")
            api_df = build_api_payload()
            save_api_outputs(api_df)
            us14_ok = verify_us14(api_df)
            logger.info(f"ANOMALY DETECTION STATUS: {'PASS' if (us13_ok and us14_ok) else 'FAIL'}")
            logger.info(f"LOG FILE: {log_path}")
            return 0 if (us13_ok and us14_ok) else 1
        except Exception:
            logger.exception("Pipeline failed")
            logger.info(f"ANOMALY DETECTION FAILED. See log: {log_path}")
            return 1
    logger.handlers[0].close()


if __name__ == "__main__":
    raise SystemExit(main())
