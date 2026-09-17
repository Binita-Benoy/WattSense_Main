"""Run US-10 feature engineering with file-only logging."""
import logging
logger = logging.getLogger()
import traceback
from logging_utils import configure_logging, capture_terminal_output
from db_loader import load_all_households
from feature_engineering.us10_features import build_feature_matrix, verify_us10


def main():
    logger, log_path = configure_logging("feature_engineering")
    with capture_terminal_output(log_path):
        try:
            logger.info("WATTSENSE — FEATURE ENGINEERING")
            df_all = load_all_households()
            features = build_feature_matrix(df_all, save=True)
            passed = verify_us10(features)
            logger.info(f"FEATURE ENGINEERING STATUS: {'PASS' if passed else 'FAIL'}")
            logger.info(f"LOG FILE: {log_path}")
            return 0 if passed else 1
        except Exception:
            logger.exception("Pipeline failed")
            logger.info(f"FEATURE ENGINEERING FAILED. See log: {log_path}")
            return 1
    logger.handlers[0].close()


if __name__ == "__main__":
    raise SystemExit(main())
