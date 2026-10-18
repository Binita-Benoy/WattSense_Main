"""Run feature engineering with terminal + file logging."""
import logging
from logging_utils import configure_logging, log_print
from db_loader import load_all_households
from feature_engineering.us10_features import build_feature_matrix, verify_us10

logger = logging.getLogger()


def main():
    logger, log_path = configure_logging("feature_engineering")
    try:
        log_print("WATTSENSE — FEATURE ENGINEERING")
        df_all = load_all_households()
        features = build_feature_matrix(df_all, save=True)
        passed = verify_us10(features)
        log_print(f"FEATURE ENGINEERING STATUS: {'PASS' if passed else 'FAIL'}")
        log_print(f"LOG FILE: {log_path}")
        return 0 if passed else 1
    except Exception:
        logger.exception("Pipeline failed")
        log_print(f"FEATURE ENGINEERING FAILED. See log: {log_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
