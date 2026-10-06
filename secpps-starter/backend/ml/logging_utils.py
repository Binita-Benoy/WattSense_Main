import logging
import os
import sys
from datetime import datetime

from config import OUT_LOGS

os.makedirs(OUT_LOGS, exist_ok=True)


def configure_logging(name: str):
    """Send pipeline messages to both the terminal and a timestamped log file."""
    log_path = os.path.join(
        OUT_LOGS,
        f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_path


def log_print(message="", level=logging.INFO):
    """Print to terminal and write the same message to the active log file."""
    print(message, flush=True)
    logging.getLogger().log(level, message)
